use core::cell::RefCell;

use num_traits::Float;

use super::DslError;
use super::ast::Definition;
use super::diff::differentiate;
use super::ir::{Graph, Node, NodeId, Ty};
use super::kernels::{EvalFn, MAX_KERNEL_ARITY, PartialFn};
use super::lower::{Lowered, lower};
use crate::dispatch::{DiffKernelCtx, GradKernelCtx, GradRef, SrcRef};
use crate::evaluate::EvalOptions;

/// Float types the DSL compiles for. Each type keeps a per-thread scratch buffer for the
/// temporary columns of step programs.
pub trait DslFloat: Float + Send + Sync + core::fmt::Debug + 'static {
    /// Runs `f` on a zero-or-garbage-filled scratch slice of length `len`. Calls must not nest.
    fn with_scratch<R>(len: usize, f: impl FnOnce(&mut [Self]) -> R) -> R;
}

macro_rules! impl_dsl_float {
    ($t:ty, $buffer:ident) => {
        thread_local! {
            static $buffer: RefCell<Vec<$t>> = const { RefCell::new(Vec::new()) };
        }

        impl DslFloat for $t {
            fn with_scratch<R>(len: usize, f: impl FnOnce(&mut [Self]) -> R) -> R {
                $buffer.with(|cell| {
                    let mut buf = cell.borrow_mut();
                    if buf.len() < len {
                        buf.resize(len, 0.0);
                    }
                    f(&mut buf[..len])
                })
            }
        }
    };
}

impl_dsl_float!(f32, SCRATCH_F32);
impl_dsl_float!(f64, SCRATCH_F64);

/// Steps skip the finiteness check; the program checks its final output once.
const STEP_OPTS: EvalOptions = EvalOptions {
    check_finite: false,
    early_exit: false,
};

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Operand<T> {
    Const(T),
    Arg(u16),
    Temp(usize),
}

#[derive(Copy, Clone, Debug)]
enum StepFn<T> {
    Eval(EvalFn<T>),
    Partial(PartialFn<T>, usize),
}

#[derive(Copy, Clone, Debug)]
struct Step<T> {
    func: StepFn<T>,
    args: [Operand<T>; MAX_KERNEL_ARITY],
    n_args: u8,
    /// `None` writes the program output.
    out: Option<usize>,
}

/// One IR root compiled to a list of kernel calls over columns. Each step writes one temporary
/// column (or the output, for the last step); temporaries are reused after their last read.
#[derive(Clone, Debug)]
pub struct Program<T> {
    steps: Vec<Step<T>>,
    /// The output when the root is a constant or an argument, which needs no steps.
    leaf: Option<Operand<T>>,
    n_temps: usize,
}

impl<T: DslFloat> Program<T> {
    pub fn compile(graph: &Graph, root: NodeId) -> Self {
        let n = root.index() + 1;
        let reachable = graph.reachable(&[root]);
        // Node ids are a topological order, so ascending ids give a valid schedule.
        let order: Vec<NodeId> = (0..n)
            .filter(|&i| reachable[i] && !graph.nodes()[i].args().is_empty())
            .map(|i| NodeId(i as u32))
            .collect();

        let mut last_use = vec![usize::MAX; n];
        for (s, id) in order.iter().enumerate() {
            for a in graph.node(*id).args() {
                last_use[a.index()] = s;
            }
        }

        let mut temp_of = vec![usize::MAX; n];
        let mut free: Vec<usize> = Vec::new();
        let mut n_temps = 0usize;
        let mut steps = Vec::with_capacity(order.len());
        for (s, &id) in order.iter().enumerate() {
            let node = graph.node(id);
            let mut args = [Operand::Const(T::zero()); MAX_KERNEL_ARITY];
            for (slot, a) in args.iter_mut().zip(node.args()) {
                *slot = operand(graph, *a, &temp_of);
            }
            let out = if id == root {
                None
            } else {
                let t = free.pop().unwrap_or_else(|| {
                    n_temps += 1;
                    n_temps - 1
                });
                temp_of[id.index()] = t;
                Some(t)
            };
            for (k, a) in node.args().iter().enumerate() {
                let first_occurrence = !node.args()[..k].contains(a);
                if first_occurrence && last_use[a.index()] == s && temp_of[a.index()] != usize::MAX {
                    free.push(temp_of[a.index()]);
                }
            }
            let func = match node {
                Node::Op { kernel, .. } => StepFn::Eval(kernel.eval_fn()),
                Node::Partial { kernel, idx, .. } => StepFn::Partial(kernel.partial_fn(), usize::from(*idx)),
                Node::Const(_) | Node::Arg(_) => unreachable!("leaves are operands, not steps"),
                Node::Call { .. } | Node::Param { .. } => unreachable!("template nodes are never compiled to steps"),
            };
            steps.push(Step {
                func,
                args,
                n_args: node.args().len() as u8,
                out,
            });
        }

        let leaf = steps.is_empty().then(|| operand(graph, root, &temp_of));
        Self { steps, leaf, n_temps }
    }

    pub fn n_steps(&self) -> usize {
        self.steps.len()
    }

    pub fn n_temps(&self) -> usize {
        self.n_temps
    }

    /// Writes the program's value for every row into `out`. `temps` must hold at least
    /// `n_temps() * out.len()` values.
    pub fn run(&self, out: &mut [T], args: &[SrcRef<'_, T>], temps: &mut [T]) {
        let n = out.len();
        if let Some(leaf) = self.leaf {
            match resolve(leaf, args, &[], 0, &[], n) {
                SrcRef::Slice(s) => out.copy_from_slice(s),
                SrcRef::Const(c) => out.fill(c),
            }
            return;
        }
        for step in &self.steps {
            let operands = &step.args[..usize::from(step.n_args)];
            let mut srcs = [SrcRef::Const(T::zero()); MAX_KERNEL_ARITY];
            match step.out {
                Some(t) => {
                    let (before, rest) = temps.split_at_mut(t * n);
                    let (dst, after) = rest.split_at_mut(n);
                    for (src, op) in srcs.iter_mut().zip(operands) {
                        *src = resolve(*op, args, before, t, after, n);
                    }
                    call(step.func, dst, &srcs[..operands.len()]);
                }
                None => {
                    let all: &[T] = temps;
                    for (src, op) in srcs.iter_mut().zip(operands) {
                        *src = resolve(*op, args, all, usize::MAX, &[], n);
                    }
                    call(step.func, out, &srcs[..operands.len()]);
                }
            }
        }
    }
}

fn operand<T: DslFloat>(graph: &Graph, id: NodeId, temp_of: &[usize]) -> Operand<T> {
    match graph.node(id) {
        Node::Const(bits) => {
            let value = f64::from_bits(*bits);
            Operand::Const(T::from(value).expect("every f64 converts to a DslFloat"))
        }
        Node::Arg(i) => Operand::Arg(*i),
        Node::Op { .. } | Node::Partial { .. } => Operand::Temp(temp_of[id.index()]),
        Node::Call { .. } | Node::Param { .. } => unreachable!("template nodes are never compiled to steps"),
    }
}

/// Resolves an operand while temporary `split` is being written: temporaries below it live in
/// `before`, those above it in `after`.
fn resolve<'a, T: Copy>(
    op: Operand<T>,
    args: &[SrcRef<'a, T>],
    before: &'a [T],
    split: usize,
    after: &'a [T],
    n: usize,
) -> SrcRef<'a, T> {
    match op {
        Operand::Const(c) => SrcRef::Const(c),
        Operand::Arg(i) => args[usize::from(i)],
        Operand::Temp(t) => {
            debug_assert_ne!(t, split, "a step never reads the temporary it writes");
            if t < split {
                SrcRef::Slice(&before[t * n..(t + 1) * n])
            } else {
                let k = t - split - 1;
                SrcRef::Slice(&after[k * n..(k + 1) * n])
            }
        }
    }
}

fn call<T>(func: StepFn<T>, out: &mut [T], args: &[SrcRef<'_, T>]) {
    match func {
        StepFn::Eval(f) => {
            f(out, args, &STEP_OPTS);
        }
        StepFn::Partial(f, idx) => f(out, args, idx),
    }
}

fn all_finite<T: Float>(values: &[T]) -> bool {
    values.iter().all(|v| v.is_finite())
}

/// A compiled DSL function: a value program and one partial-derivative program per argument.
#[derive(Clone, Debug)]
pub struct ScalarFn<T> {
    arity: u8,
    returns_bool: bool,
    value: Program<T>,
    partials: Vec<Program<T>>,
    /// The most temporaries any one program needs.
    max_temps: usize,
}

impl<T: DslFloat> ScalarFn<T> {
    #[doc(hidden)]
    pub fn from_lowered(lowered: &Lowered) -> Self {
        let mut graph = lowered.graph.clone();
        let partial_roots: Vec<NodeId> = (0..u16::from(lowered.arity))
            .map(|k| differentiate(&mut graph, lowered.root, k))
            .collect();
        let value = Program::compile(&graph, lowered.root);
        let partials: Vec<Program<T>> = partial_roots.iter().map(|&r| Program::compile(&graph, r)).collect();
        let max_temps = partials.iter().map(Program::n_temps).fold(value.n_temps(), usize::max);
        Self {
            arity: lowered.arity,
            returns_bool: lowered.ty == Ty::Bool,
            value,
            partials,
            max_temps,
        }
    }

    /// Compiles a standalone definition, such as an elementwise loss or an early-stop condition.
    /// Its body may call built-in kernels.
    pub fn compile(def: &Definition) -> Result<Self, DslError> {
        Ok(Self::from_lowered(&lower(def, &[])?))
    }

    pub fn arity(&self) -> u8 {
        self.arity
    }

    pub fn returns_bool(&self) -> bool {
        self.returns_bool
    }

    #[doc(hidden)]
    pub fn value_program(&self) -> &Program<T> {
        &self.value
    }

    #[doc(hidden)]
    pub fn partial_program(&self, arg: usize) -> &Program<T> {
        &self.partials[arg]
    }

    /// Evaluates the function row-wise into `out`, with the same finiteness rule as
    /// [`crate::evaluate::kernels::eval_nary`].
    pub fn eval(&self, out: &mut [T], args: &[SrcRef<'_, T>], opts: &EvalOptions) -> bool {
        debug_assert_eq!(args.len(), usize::from(self.arity));
        let n = out.len();
        T::with_scratch(self.value.n_temps() * n, |temps| self.value.run(out, args, temps));
        if !opts.check_finite {
            return true;
        }
        let finite = all_finite(out);
        if !finite && opts.early_exit {
            out.fill(T::nan());
        }
        finite
    }

    /// Writes the partial derivative with respect to argument `arg` into `out`, row-wise.
    pub fn partial(&self, arg: usize, out: &mut [T], args: &[SrcRef<'_, T>]) {
        debug_assert_eq!(args.len(), usize::from(self.arity));
        let program = &self.partials[arg];
        T::with_scratch(program.n_temps() * out.len(), |temps| program.run(out, args, temps));
    }

    /// Forward-mode derivative along `ctx.dargs`, with the same finiteness rule as
    /// [`crate::evaluate::kernels::diff_nary`].
    pub fn diff(&self, ctx: DiffKernelCtx<'_, '_, T>) -> bool {
        debug_assert_eq!(ctx.args.len(), usize::from(self.arity));
        debug_assert_eq!(ctx.dargs.len(), usize::from(self.arity));
        let n = ctx.out_val.len();
        T::with_scratch((self.max_temps + 1) * n, |buf| {
            let (partial, temps) = buf.split_at_mut(n);
            self.value.run(ctx.out_val, ctx.args, temps);
            ctx.out_der.fill(T::zero());
            for (program, darg) in self.partials.iter().zip(ctx.dargs) {
                if matches!(darg, SrcRef::Const(c) if c.is_zero()) {
                    continue;
                }
                program.run(partial, ctx.args, temps);
                match *darg {
                    SrcRef::Slice(d) => {
                        for ((o, &p), &dv) in ctx.out_der.iter_mut().zip(partial.iter()).zip(d) {
                            *o = *o + p * dv;
                        }
                    }
                    SrcRef::Const(c) => {
                        for (o, &p) in ctx.out_der.iter_mut().zip(partial.iter()) {
                            *o = *o + p * c;
                        }
                    }
                }
            }
        });

        if !ctx.opts.check_finite {
            return true;
        }
        let finite = all_finite(ctx.out_val) && (self.arity <= 2 || all_finite(ctx.out_der));
        if !finite && ctx.opts.early_exit {
            ctx.out_val.fill(T::nan());
            ctx.out_der.fill(T::nan());
        }
        finite
    }

    /// Gradient along `ctx.n_dir` directions, with the same finiteness rule as
    /// [`crate::evaluate::kernels::grad_nary`].
    pub fn grad(&self, ctx: GradKernelCtx<'_, '_, T>) -> bool {
        debug_assert_eq!(ctx.args.len(), usize::from(self.arity));
        debug_assert_eq!(ctx.arg_grads.len(), usize::from(self.arity));
        let n = ctx.n_rows;
        let grads = &mut ctx.out_grad[..ctx.n_dir * n];
        T::with_scratch((self.max_temps + 1) * n, |buf| {
            let (partial, temps) = buf.split_at_mut(n);
            self.value.run(ctx.out_val, ctx.args, temps);
            grads.fill(T::zero());
            for (program, arg_grad) in self.partials.iter().zip(ctx.arg_grads) {
                if matches!(arg_grad, GradRef::Zero) {
                    continue;
                }
                program.run(partial, ctx.args, temps);
                match *arg_grad {
                    GradRef::Basis(k) if k < ctx.n_dir => {
                        for (o, &p) in grads[k * n..(k + 1) * n].iter_mut().zip(partial.iter()) {
                            *o = *o + p;
                        }
                    }
                    GradRef::Slice(s) => {
                        for (g_dir, s_dir) in grads.chunks_mut(n).zip(s.chunks(n)) {
                            for ((o, &p), &sv) in g_dir.iter_mut().zip(partial.iter()).zip(s_dir) {
                                *o = *o + p * sv;
                            }
                        }
                    }
                    GradRef::Basis(_) | GradRef::Zero => {}
                }
            }
        });

        if !ctx.opts.check_finite {
            return true;
        }
        let finite = all_finite(ctx.out_val);
        if !finite && ctx.opts.early_exit {
            ctx.out_val.fill(T::nan());
            ctx.out_grad.fill(T::nan());
        }
        finite
    }
}
