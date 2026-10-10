use std::str::FromStr;
use std::sync::Arc;

use dynamic_expressions::Operators;
use dynamic_expressions::dispatch::SrcRef;
use dynamic_expressions::dsl::{DslError, DslFloat, ScalarFn, parse_definition};
use num_traits::Float;

use crate::loss_functions::{BuiltinLoss, LossObject};

/// Matches SymbolicRegression.jl `early_stop_condition`: takes `(loss, complexity)` and returns
/// `true` to terminate the search early. Wrapping a scalar threshold `t` is `EarlyStop::below(t)`.
pub type EarlyStopCondition<T> = Arc<dyn Fn(T, usize) -> bool + Send + Sync>;

/// Helper constructors for the standard kinds of early stop conditions Julia supports.
pub struct EarlyStop;

impl EarlyStop {
    /// Stop when any hall-of-fame member has `loss < threshold`. Matches Julia's wrapping of a
    /// scalar `early_stop_condition::Real` into `(l, c) -> l < threshold`.
    pub fn below<T: Float + Send + Sync + 'static>(threshold: T) -> EarlyStopCondition<T> {
        Arc::new(move |loss: T, _complexity: usize| loss < threshold)
    }
}

/// Parses an early-stop condition written as in PySR's `early_stop_condition`: a loss threshold
/// such as `"1e-6"` (stop once a loss is below it), or a definition taking `(loss, complexity)`
/// and returning Bool, such as `"stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"`.
///
/// ```
/// use symbolic_regression::parse_early_stop;
///
/// let stop = parse_early_stop::<f64>("(loss, complexity) -> loss <= 1e-4 && complexity <= 6").unwrap();
/// assert!(stop(1e-5, 6));
/// assert!(!stop(1e-5, 7));
/// assert!(parse_early_stop::<f64>("1e-6").unwrap()(1e-7, 30));
/// ```
pub fn parse_early_stop<T: DslFloat>(src: &str) -> Result<EarlyStopCondition<T>, DslError> {
    if let Ok(threshold) = src.trim().parse::<f64>() {
        return Ok(EarlyStop::below(T::from(threshold).unwrap()));
    }
    let def = parse_definition(src)?;
    let n = def.params.len();
    if n != 2 {
        let plural = if n == 1 { "" } else { "s" };
        return Err(DslError::new(
            format!("an early-stop condition takes (loss, complexity), but this one takes {n} argument{plural}"),
            def.span,
            src,
        ));
    }
    let func = ScalarFn::<T>::compile(&def)?;
    if !func.returns_bool() {
        return Err(DslError::new(
            "an early-stop condition must return Bool, but this one returns a number",
            def.span,
            src,
        ));
    }
    let opts = dynamic_expressions::EvalOptions {
        check_finite: false,
        early_exit: false,
    };
    Ok(Arc::new(move |loss: T, complexity: usize| {
        let mut out = [T::zero()];
        let args = [SrcRef::Const(loss), SrcRef::Const(T::from(complexity).unwrap())];
        func.eval(&mut out, &args, &opts);
        out[0] != T::zero()
    }))
}

/// How SymbolicRegression.jl scores the Pareto front (`loss_scale`).
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum LossScale {
    /// Scores are the drop in log-loss per unit of complexity; losses must be non-negative.
    #[default]
    Log,
    /// Scores are the drop in loss per unit of complexity; losses may be negative, and the best
    /// equation is the one with the lowest loss.
    Linear,
}

impl FromStr for LossScale {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "log" => Ok(Self::Log),
            "linear" => Ok(Self::Linear),
            other => Err(format!("`loss_scale` must be either log or linear, got `{other}`")),
        }
    }
}

#[rustfmt::skip]
macro_rules! sr_mutation_weights_spec {
    ($m:ident) => {
        $m! {
            mutate_constant:
                (f64, 0.0346, "mw-mutate-constant"),
            mutate_operator:
                (f64, 0.293, "mw-mutate-operator"),
            mutate_feature:
                (f64, 0.1, "mw-mutate-feature"),
            swap_operands:
                (f64, 0.198, "mw-swap-operands"),
            rotate_tree:
                (f64, 4.26, "mw-rotate-tree"),
            add_node:
                (f64, 2.47, "mw-add-node"),
            insert_node:
                (f64, 0.0112, "mw-insert-node"),
            delete_node:
                (f64, 0.870, "mw-delete-node"),
            simplify:
                (f64, 0.00209, "mw-simplify"),
            randomize:
                (f64, 0.000502, "mw-randomize"),
            do_nothing:
                (f64, 0.273, "mw-do-nothing"),
            optimize:
                (f64, 0.0, "mw-optimize"),
            form_connection:
                (f64, 0.5, "mw-form-connection"),
            break_connection:
                (f64, 0.1, "mw-break-connection"),
        }
    };
}

#[rustfmt::skip]
macro_rules! sr_options_spec {
    ($m:ident) => {
        $m! {
            values {
                seed:
                    (u64, 0, "seed"),
                niterations:
                    (usize, 100, "niterations"),
                populations:
                    (usize, 31, "populations"),
                population_size:
                    (usize, 27, "population-size"),
                ncycles_per_iteration:
                    (usize, 380, "ncycles-per-iteration"),
                batch_size:
                    (usize, 50, "batch-size"),
                complexity_of_constants:
                    (f32, 1.0, "complexity-of-constants"),
                complexity_of_variables:
                    (f32, 1.0, "complexity-of-variables"),
                maxsize:
                    (usize, 30, "maxsize"),
                maxdepth:
                    (usize, 30, "maxdepth"),
                warmup_maxsize_by:
                    (f32, 0.0, "warmup-maxsize-by"),
                parsimony:
                    (f64, 0.0, "parsimony"),
                adaptive_parsimony_scaling:
                    (f64, 20.0, "adaptive-parsimony-scaling"),
                crossover_probability:
                    (f64, 0.0259, "crossover-probability"),
                perturbation_factor:
                    (f64, 0.129, "perturbation-factor"),
                probability_negate_constant:
                    (f64, 0.00743, "probability-negate-constant"),
                tournament_selection_n:
                    (usize, 15, "tournament-selection-n"),
                tournament_selection_p:
                    (f32, 0.982, "tournament-selection-p"),
                alpha:
                    (f64, 3.17, "alpha"),
                optimizer_nrestarts:
                    (usize, 2, "optimizer-nrestarts"),
                optimizer_probability:
                    (f64, 0.14, "optimizer-probability"),
                optimizer_iterations:
                    (usize, 8, "optimizer-iterations"),
                optimizer_f_calls_limit:
                    (usize, 10_000, "optimizer-f-calls-limit"),
                fraction_replaced:
                    (f64, 0.00036, "fraction-replaced"),
                fraction_replaced_hof:
                    (f64, 0.0614, "fraction-replaced-hof"),
                topn:
                    (usize, 12, "topn"),
                print_precision:
                    (usize, 5, "print-precision"),
                max_evals:
                    (u64, 0, "max-evals"),
                timeout_in_seconds:
                    (f64, 0.0, "timeout-in-seconds"),
                loss_scale:
                    (LossScale, LossScale::Log, "loss-scale"),
            }
            neg_flags {
                use_frequency:
                    (true, no_use_frequency, "no-use-frequency"),
                use_frequency_in_tournament:
                    (true, no_use_frequency_in_tournament, "no-use-frequency-in-tournament"),
                skip_mutation_failures:
                    (true, no_skip_mutation_failures, "no-skip-mutation-failures"),
                annealing:
                    (true, no_annealing, "no-annealing"),
                should_optimize_constants:
                    (true, no_should_optimize_constants, "no-should-optimize-constants"),
                migration:
                    (true, no_migration, "no-migration"),
                hof_migration:
                    (true, no_hof_migration, "no-hof-migration"),
                use_baseline:
                    (true, no_use_baseline, "no-use-baseline"),
                progress:
                    (true, no_progress, "no-progress"),
            }
            pos_flags {
                should_simplify:
                    (true, should_simplify, "should-simplify"),
                batching:
                    (false, batching, "batching"),
                deterministic:
                    (false, deterministic, "deterministic"),
            }
        }
    };
}

#[rustfmt::skip]
macro_rules! __define_mutation_weights {
    ( $( $name:ident: ($ty:ty, $default:expr, $cli_long:literal), )* ) => {
        #[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
        #[cfg_attr(feature = "serde", serde(default))]
        #[derive(Clone, Debug)]
        pub struct MutationWeights {
            $(pub $name: $ty,)*
        }

        impl Default for MutationWeights {
            fn default() -> Self {
                Self { $($name: $default,)* }
            }
        }
    };
}

sr_mutation_weights_spec!(__define_mutation_weights);

#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum OutputStyle {
    /// Enable ANSI styles only when stderr supports it (and `NO_COLOR` is not set).
    #[default]
    Auto,
    /// Disable ANSI styles.
    Plain,
    /// Force ANSI styles (even when stderr is not a TTY).
    Ansi,
}

macro_rules! __define_options {
    (
        values { $( $name:ident: ($ty:ty, $default:expr, $cli_long:literal), )* }
        neg_flags { $( $iname:ident: ($bdefault:expr, $cli_name:ident, $cli_blong:literal), )* }
        pos_flags { $( $pname:ident: ($pdefault:expr, $cli_pname:ident, $cli_plong:literal), )* }
    ) => {
        #[derive(Clone)]
        pub struct Options<T: Float, Ops, const D: usize> {
            $(pub $name: $ty,)*
            $(pub $iname: bool,)*
            $(pub $pname: bool,)*

            /// The operator set that evaluates, prints, and parses every expression in the
            /// search (`operators.set`), and the operators the search may use. Operators written
            /// as strings live in the set's `custom` table and get ids after the built-ins:
            ///
            /// ```
            /// use symbolic_regression::prelude::*;
            ///
            /// let (custom, names) = parse_operator_tokens::<f64, _>(["+", "*", "inv(x) = 1/x"]).unwrap();
            /// let options = Options::<f64, _, 3> {
            ///     operators: BuiltinOpsF64::new().with_custom(custom).from_names(&names).unwrap(),
            ///     ..Default::default()
            /// };
            /// assert_eq!(options.operators.nops(1), 1);
            /// assert_eq!(options.operators.nops(2), 2);
            /// ```
            pub operators: Operators<Ops, D>,
            pub mutation_weights: MutationWeights,
            /// The loss each equation minimizes; L2 (`L2DistLoss()`) by default. Losses written as
            /// strings, as PySR's `elementwise_loss` takes them, come from
            /// [`parse_loss`](crate::parse_loss):
            ///
            /// ```
            /// use symbolic_regression::{Options, parse_loss};
            /// use symbolic_regression::prelude::BuiltinOpsF64;
            ///
            /// let options = Options::<f64, BuiltinOpsF64, 2> {
            ///     loss: parse_loss("myloss(x, y) = abs(x - y)^1.5").unwrap(),
            ///     ..Default::default()
            /// };
            /// assert_eq!(options.loss.loss(&[3.0], &[-1.0], None), 8.0);
            /// ```
            pub loss: LossObject<T>,

            pub output_style: OutputStyle,

            /// Complexity costs may be fractional; a tree's complexity is the sum of its node
            /// costs rounded to the nearest integer, as in SymbolicRegression.jl.
            pub variable_complexities: Option<Vec<f32>>,
            pub operator_complexity_overrides: std::collections::HashMap<
                dynamic_expressions::OpId,
                f32,
            >,
            pub op_constraints: crate::check_constraints::OpConstraints<D>,
            pub nested_constraints: crate::check_constraints::NestedConstraints,

            /// Optional early-stop callback. Matches SymbolicRegression.jl `early_stop_condition`:
            /// the search stops as soon as any hall-of-fame member satisfies `f(loss, complexity)`.
            /// [`parse_early_stop`] reads PySR-style strings.
            pub early_stop_condition: Option<EarlyStopCondition<T>>,
        }

        impl<T: Float, Ops: Default, const D: usize> Default for Options<T, Ops, D> {
            fn default() -> Self {
                Self {
                    $($name: $default,)*
                    $($iname: $bdefault,)*
                    $($pname: $pdefault,)*
                    operators: Operators::default(),
                    mutation_weights: MutationWeights::default(),
                    loss: BuiltinLoss::L2Dist.into_loss(),
                    output_style: OutputStyle::Auto,
                    variable_complexities: None,
                    operator_complexity_overrides: std::collections::HashMap::new(),
                    op_constraints: Default::default(),
                    nested_constraints: Default::default(),
                    early_stop_condition: None,
                }
            }
        }
    };
}

sr_options_spec!(__define_options);

macro_rules! __define_wasm_options_shim {
    (
        values { $( $name:ident: ($ty:ty, $default:expr, $cli_long:literal), )* }
        neg_flags { $( $iname:ident: ($bdefault:expr, $cli_name:ident, $cli_blong:literal), )* }
        pos_flags { $( $pname:ident: ($pdefault:expr, $cli_pname:ident, $cli_plong:literal), )* }
    ) => {
        /// Serde-friendly "wire-format" representation of the SR engine's tunable knobs.
        ///
        /// This contains only the option fields listed in `sr_options_spec!` plus
        /// `mutation_weights`, and can be applied to an `Options` instance.
        #[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
        #[cfg_attr(feature = "serde", serde(default))]
        #[derive(Clone, Debug)]
        pub struct WasmOptionsShim {
            $(pub $name: $ty,)*
            $(pub $iname: bool,)*
            $(pub $pname: bool,)*

            pub mutation_weights: MutationWeights,
        }

        impl Default for WasmOptionsShim {
            fn default() -> Self {
                Self {
                    $($name: $default,)*
                    $($iname: $bdefault,)*
                    $($pname: $pdefault,)*
                    mutation_weights: MutationWeights::default(),
                }
            }
        }

        impl WasmOptionsShim {
            pub fn apply_to<T: Float, Ops, const D: usize>(&self, opt: &mut Options<T, Ops, D>) {
                $(opt.$name = self.$name;)*
                $(opt.$iname = self.$iname;)*
                $(opt.$pname = self.$pname;)*
                opt.mutation_weights = self.mutation_weights.clone();
            }
        }
    };
}

sr_options_spec!(__define_wasm_options_shim);

impl<T: Float, Ops: dynamic_expressions::OperatorSet, const D: usize> Options<T, Ops, D> {
    #[inline]
    pub fn uses_default_complexity(&self) -> bool {
        self.complexity_of_constants == 1.0
            && self.complexity_of_variables == 1.0
            && self.variable_complexities.is_none()
            && self.operator_complexity_overrides.is_empty()
            && (Ops::UNIT_COMPLEXITY
                || self
                    .operators
                    .ops_by_arity
                    .iter()
                    .flatten()
                    .all(|&op| self.operators.set.complexity(op) == 1))
    }
}
