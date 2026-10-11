use std::str::FromStr;
use std::sync::Arc;

use dynamic_expressions::Operators;
use dynamic_expressions::dispatch::SrcRef;
use dynamic_expressions::dsl::{DslError, DslFloat, ScalarFn, TemplateStructure, parse_definition};
use num_traits::Float;

use crate::backsolve::BacksolveOptions;
use crate::dataset::Dataset;
use crate::guesses::Guess;
use crate::loss_functions::{BuiltinLoss, LossObject};
use crate::plugins::{AdaptiveMutationWeights, MutationBurst};

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
/// written as a Julia number, such as `"1e-6"` or `"1f-6"` (stop once a loss is below it), or a
/// definition taking `(loss, complexity)` and returning Bool, such as
/// `"stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"`.
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
    if let Some(threshold) = dynamic_expressions::dsl::lexer::number_literal(src) {
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

/// Hyperparameter profiles selected by SymbolicRegression.jl's defaults version.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum DefaultsProfile {
    Legacy,
    V1,
    #[default]
    V2,
}

/// Whether a search evolves on sampled rows.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum Batching {
    #[default]
    Auto,
    On,
    Off,
}

/// Constant optimizer used when an expression has more than one parameter.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum OptimizerAlgorithm {
    #[default]
    BFGS,
    NelderMead,
}

/// Execution mode for independent populations.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum Parallelism {
    Serial,
    #[default]
    Multithreading,
}

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ConstantMutation {
    pub perturbation_factor: f64,
    pub probability_negate: f64,
}
impl Default for ConstantMutation {
    fn default() -> Self {
        Self {
            perturbation_factor: 0.129,
            probability_negate: 0.00743,
        }
    }
}

/// Built-in mutation configuration, with weights supplied separately.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BuiltinMutation {
    Constant(ConstantMutation),
    Operator,
    Feature,
    SwapOperands,
    RotateTree,
    AddNode,
    InsertNode,
    DeleteNode,
    Simplify,
    Randomize,
    DoNothing,
    Optimize,
    Backsolve(BacksolveOptions),
}
impl BuiltinMutation {
    pub(crate) fn same_kind(self, other: Self) -> bool {
        std::mem::discriminant(&self) == std::mem::discriminant(&other)
    }
}

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Crossover {
    Subtree,
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
            backsolve:
                (f64, 0.0, "mw-backsolve"),
        }
    };
}

#[rustfmt::skip]
macro_rules! sr_options_spec {
    ($m:ident) => {
        $m! {
            values {
                batching:
                    (Batching, Batching::Auto, "batching"),
                should_simplify:
                    (Option<bool>, None, "should-simplify"),
                optimizer_algorithm:
                    (OptimizerAlgorithm, OptimizerAlgorithm::BFGS, "optimizer-algorithm"),
                optimizer_g_abstol:
                    (f64, 1e-8, "optimizer-g-abstol"),
                optimizer_x_abstol:
                    (f64, 0.0, "optimizer-x-abstol"),
                optimizer_f_reltol:
                    (f64, 0.0, "optimizer-f-reltol"),
                optimizer_g_calls_limit:
                    (usize, 0, "optimizer-g-calls-limit"),
                parallelism:
                    (Parallelism, Parallelism::Multithreading, "parallelism"),
                seed:
                    (Option<u64>, None, "seed"),
                niterations:
                    (usize, 100, "niterations"),
                populations:
                    (usize, 31, "populations"),
                population_size:
                    (usize, 27, "population-size"),
                ncycles_per_iteration:
                    (usize, 380, "ncycles-per-iteration"),
                batch_size:
                    (Option<usize>, None, "batch-size"),
                complexity_of_constants:
                    (f32, 1.0, "complexity-of-constants"),
                complexity_of_variables:
                    (f32, 1.0, "complexity-of-variables"),
                maxsize:
                    (usize, 30, "maxsize"),
                maxdepth:
                    (Option<usize>, None, "maxdepth"),
                warmup_maxsize_by:
                    (f32, 0.0, "warmup-maxsize-by"),
                parsimony:
                    (f64, 0.0, "parsimony"),
                adaptive_parsimony_scaling:
                    (f64, 1040.0, "adaptive-parsimony-scaling"),
                crossover_probability:
                    (f64, 0.20, "crossover-probability"),
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
                fraction_replaced_guesses:
                    (f64, 0.001, "fraction-replaced-guesses"),
                topn:
                    (usize, 12, "topn"),
                verbosity:
                    (Option<i32>, None, "verbosity"),
                terminal_width:
                    (Option<usize>, None, "terminal-width"),
                print_precision:
                    (usize, 5, "print-precision"),
                max_evals:
                    (u64, 0, "max-evals"),
                timeout_in_seconds:
                    (f64, 0.0, "timeout-in-seconds"),
                loss_scale:
                    (LossScale, LossScale::Log, "loss-scale"),
                dimensional_constraint_penalty:
                    (f64, 1000.0, "dimensional-constraint-penalty"),
            }
            neg_flags {
                save_to_file:
                    (true, no_save_to_file, "no-save-to-file"),
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
                deterministic:
                    (false, deterministic, "deterministic"),
                dimensionless_constants_only:
                    (false, dimensionless_constants_only, "dimensionless-constants-only"),
                use_tracing:
                    (false, use_tracing, "use-tracing"),
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
            /// Parameters of the backsolve mutation (SymbolicRegression.jl's `BacksolveMutation`),
            /// which fires with weight `mutation_weights.backsolve`.
            pub backsolve: BacksolveOptions,
            pub mutations: Vec<(BuiltinMutation, f64)>,
            pub default_mutations: Option<Vec<(BuiltinMutation, f64)>>,
            pub crossovers: Vec<(Crossover, f64)>,
            pub default_crossovers: Option<Vec<(Crossover, f64)>>,
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
            /// The base directory for hall-of-fame files; `None` means `outputs`.
            pub output_directory: Option<String>,
            /// A run identifier; `None` generates a timestamp and six alphanumeric characters.
            pub run_id: Option<String>,
            pub logger: Option<crate::run_output::SrLogger<T>>,
            /// Nonblocking user input. `None` disables quit-command monitoring.
            pub input_stream: Option<crate::stop_controller::InputStream>,
            /// A per-search graceful stop request, checked after completed population cycles.
            pub external_stop: Option<crate::stop_controller::ExternalStop>,

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

            /// SymbolicRegression.jl's `expression_spec=TemplateExpressionSpec(...)`: the search
            /// evolves one tree per subexpression of the template, and its parameter vectors.
            /// Category-indexed parameters (`p[class]`) replace SymbolicRegression.jl's former
            /// `ParametricExpression`:
            ///
            /// ```
            /// use symbolic_regression::Options;
            /// use symbolic_regression::prelude::{BuiltinOpsF64, TemplateStructure};
            ///
            /// let template = TemplateStructure::new(
            ///     "f(x1, x2, p1[category], p2[category])",
            ///     &["f"],
            ///     None,
            ///     &["x1", "x2", "category"],
            ///     &[("p1", 3), ("p2", 3)],
            ///     &BuiltinOpsF64::new(),
            /// )
            /// .unwrap();
            /// assert_eq!(template.num_features(), &[4]);
            /// let options = Options::<f64, BuiltinOpsF64, 2> {
            ///     template: Some(template),
            ///     ..Default::default()
            /// };
            /// ```
            pub template: Option<TemplateStructure>,

            /// SymbolicRegression.jl's `guesses`: one list of equations per output that seed the
            /// search (see [`Guess`]). Each enters the hall of fame with its constants optimized,
            /// and after every cycle a population replaces a `fraction_replaced_guesses` share of
            /// its members with them on average.
            ///
            /// ```
            /// use symbolic_regression::{Guess, Options};
            /// use symbolic_regression::prelude::BuiltinOpsF64;
            ///
            /// let options = Options::<f64, BuiltinOpsF64, 2> {
            ///     guesses: vec![vec![Guess::from("2.0*x0*x0 + 3.0*x1 + 0.5")]],
            ///     ..Default::default()
            /// };
            /// ```
            pub guesses: Vec<Vec<Guess<T, Ops, D>>>,

            /// SymbolicRegression.jl's `tracing_file`: the JSON Lines file a search with
            /// `use_tracing` writes its trace to, replacing any earlier contents.
            pub tracing_file: std::path::PathBuf,
            /// SymbolicRegression.jl's default `AdaptiveMutationWeightsPlugin`; `None` samples
            /// the configured `mutation_weights` throughout.
            pub adaptive_mutation_weights: Option<AdaptiveMutationWeights>,
            /// SymbolicRegression.jl's `MutationBurstPlugin`, off by default.
            pub mutation_burst: Option<MutationBurst>,
        }

        impl<T: Float, Ops: Default, const D: usize> Default for Options<T, Ops, D> {
            fn default() -> Self {
                Self {
                    $($name: $default,)*
                    $($iname: $bdefault,)*
                    $($pname: $pdefault,)*
                    operators: Operators::default(),
                    mutation_weights: MutationWeights::default(),
                    backsolve: BacksolveOptions::default(),
                    mutations: Vec::new(),
                    default_mutations: None,
                    crossovers: Vec::new(),
                    default_crossovers: None,
                    loss: BuiltinLoss::L2Dist.into_loss(),
                    output_style: OutputStyle::Auto,
                    output_directory: None,
                    run_id: None,
                    logger: None,
                    input_stream: crate::stop_controller::InputStream::default_stdin(),
                    external_stop: None,
                    variable_complexities: None,
                    operator_complexity_overrides: std::collections::HashMap::new(),
                    op_constraints: Default::default(),
                    nested_constraints: Default::default(),
                    early_stop_condition: None,
                    template: None,
                    guesses: Vec::new(),
                    tracing_file: "pysr_trace.jsonl".into(),
                    adaptive_mutation_weights: Some(AdaptiveMutationWeights::default()),
                    mutation_burst: None,
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
            pub mutations: Vec<(BuiltinMutation, f64)>,
            pub default_mutations: Option<Vec<(BuiltinMutation, f64)>>,
            pub crossovers: Vec<(Crossover, f64)>,
            pub default_crossovers: Option<Vec<(Crossover, f64)>>,
        }

        impl Default for WasmOptionsShim {
            fn default() -> Self {
                Self {
                    $($name: $default,)*
                    $($iname: $bdefault,)*
                    $($pname: $pdefault,)*
                    mutation_weights: MutationWeights::default(),
                    mutations: Vec::new(),
                    default_mutations: None,
                    crossovers: Vec::new(),
                    default_crossovers: None,
                }
            }
        }

        impl WasmOptionsShim {
            pub fn apply_to<T: Float, Ops, const D: usize>(&self, opt: &mut Options<T, Ops, D>) {
                $(opt.$name = self.$name;)*
                $(opt.$iname = self.$iname;)*
                $(opt.$pname = self.$pname;)*
                opt.mutation_weights = self.mutation_weights.clone();
                opt.mutations.clone_from(&self.mutations);
                opt.default_mutations.clone_from(&self.default_mutations);
                opt.crossovers.clone_from(&self.crossovers);
                opt.default_crossovers.clone_from(&self.default_crossovers);
            }
        }
    };
}

sr_options_spec!(__define_wasm_options_shim);

macro_rules! __define_options_describe {
    (
        values { $( $name:ident: ($ty:ty, $default:expr, $cli_long:literal), )* }
        neg_flags { $( $iname:ident: ($bdefault:expr, $cli_name:ident, $cli_blong:literal), )* }
        pos_flags { $( $pname:ident: ($pdefault:expr, $cli_pname:ident, $cli_plong:literal), )* }
    ) => {
        impl<T: Float, Ops, const D: usize> Options<T, Ops, D> {
            /// SymbolicRegression.jl's `string(options)`: `Options(` followed by each setting
            /// as `name=value`.
            pub(crate) fn describe(&self) -> String {
                let fields = [
                    $(format!("{}={:?}", stringify!($name), self.$name),)*
                    $(format!("{}={:?}", stringify!($iname), self.$iname),)*
                    $(format!("{}={:?}", stringify!($pname), self.$pname),)*
                    format!("mutation_weights={:?}", self.mutation_weights),
                    format!("tracing_file={:?}", self.tracing_file),
                ];
                format!("Options({})", fields.join(", "))
            }
        }
    };
}

sr_options_spec!(__define_options_describe);
impl<T: Float, Ops: Default, const D: usize> Options<T, Ops, D> {
    pub fn with_defaults(profile: DefaultsProfile) -> Self {
        let mut options = Self::default();
        if profile != DefaultsProfile::V2 {
            options.crossover_probability = 0.0259;
            options.batching = Batching::Off;
            options.batch_size = Some(50);
        }
        if profile == DefaultsProfile::Legacy {
            options.maxsize = 20;
            options.populations = 15;
            options.population_size = 33;
            options.ncycles_per_iteration = 550;
            options.parsimony = 0.0032;
            options.adaptive_parsimony_scaling = 20.0;
            options.mutation_weights = MutationWeights {
                mutate_constant: 0.048,
                mutate_operator: 0.47,
                swap_operands: 0.1,
                rotate_tree: 0.0,
                add_node: 0.79,
                insert_node: 5.1,
                delete_node: 1.7,
                simplify: 0.0020,
                randomize: 0.00023,
                do_nothing: 0.21,
                ..Default::default()
            };
            options.crossover_probability = 0.066;
            options.annealing = false;
            options.alpha = 0.1;
            options.perturbation_factor = 0.076;
            options.probability_negate_constant = 0.01;
            options.tournament_selection_n = 12;
            options.tournament_selection_p = 0.86;
            options.fraction_replaced_hof = 0.035;
        }
        options
    }
}

impl<T: Float, Ops, const D: usize> Options<T, Ops, D> {
    #[inline]
    pub fn effective_maxdepth(&self) -> usize {
        self.maxdepth.unwrap_or(self.maxsize)
    }

    #[inline]
    pub fn use_batching(&self, n: usize) -> bool {
        self.batching == Batching::On || (self.batching == Batching::Auto && n > 1000)
    }

    #[inline]
    pub fn get_batch_size(&self, n: usize) -> usize {
        self.batch_size.map_or_else(
            || match n {
                0..=1000 => n,
                1001..5000 => 128,
                5000..50000 => 256,
                _ => 512,
            },
            |size| size.min(n),
        )
    }

    #[inline]
    pub fn batching_required(&self, n: usize) -> bool {
        self.use_batching(n) && self.get_batch_size(n) < n
    }

    #[inline]
    pub fn effective_should_simplify(&self) -> bool {
        self.should_simplify
            .unwrap_or(self.op_constraints.limits.is_empty() && self.nested_constraints.limits.is_empty())
    }

    pub fn default_mutations(&self) -> [(BuiltinMutation, f64); 13] {
        let w = &self.mutation_weights;
        [
            (
                BuiltinMutation::Constant(ConstantMutation {
                    perturbation_factor: self.perturbation_factor,
                    probability_negate: self.probability_negate_constant,
                }),
                w.mutate_constant,
            ),
            (BuiltinMutation::Operator, w.mutate_operator),
            (BuiltinMutation::Feature, w.mutate_feature),
            (BuiltinMutation::SwapOperands, w.swap_operands),
            (BuiltinMutation::RotateTree, w.rotate_tree),
            (BuiltinMutation::AddNode, w.add_node),
            (BuiltinMutation::InsertNode, w.insert_node),
            (BuiltinMutation::DeleteNode, w.delete_node),
            (BuiltinMutation::Simplify, w.simplify),
            (BuiltinMutation::Randomize, w.randomize),
            (BuiltinMutation::DoNothing, w.do_nothing),
            (BuiltinMutation::Optimize, w.optimize),
            (BuiltinMutation::Backsolve(self.backsolve), w.backsolve),
        ]
    }

    pub fn resolved_mutations(&self) -> impl Iterator<Item = (BuiltinMutation, f64)> + '_ {
        self.mutations.iter().copied().chain(
            self.default_mutations
                .iter()
                .flatten()
                .copied()
                .chain(
                    self.default_mutations()
                        .into_iter()
                        .filter(|_| self.default_mutations.is_none()),
                )
                .filter(|(mutation, _)| !self.mutations.iter().any(|(explicit, _)| explicit.same_kind(*mutation))),
        )
    }

    pub fn resolved_crossovers(&self) -> impl Iterator<Item = (Crossover, f64)> + '_ {
        self.crossovers.iter().copied().chain(
            self.default_crossovers
                .iter()
                .flatten()
                .copied()
                .chain(
                    [(Crossover::Subtree, 1.0)]
                        .into_iter()
                        .filter(|_| self.default_crossovers.is_none()),
                )
                .filter(|(crossover, _)| !self.crossovers.iter().any(|(explicit, _)| explicit == crossover)),
        )
    }

    pub(crate) fn constant_mutation(&self) -> ConstantMutation {
        ConstantMutation {
            perturbation_factor: self.perturbation_factor,
            probability_negate: self.probability_negate_constant,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.maxsize <= 3 {
            return Err("maxsize must be greater than 3".into());
        }
        if self.warmup_maxsize_by.is_nan() || self.warmup_maxsize_by < 0.0 {
            return Err("warmup_maxsize_by must be non-negative".into());
        }
        if self.tournament_selection_n >= self.population_size {
            return Err("tournament_selection_n must be less than population_size".into());
        }
        if self.batch_size == Some(0) {
            return Err("batch_size must be positive".into());
        }
        for (_, weight) in self.resolved_mutations() {
            if !weight.is_finite() || weight < 0.0 {
                return Err("mutation weights must be finite and non-negative".into());
            }
        }
        for (_, weight) in self.resolved_crossovers() {
            if !weight.is_finite() || weight < 0.0 {
                return Err("crossover weights must be finite and non-negative".into());
            }
        }
        Ok(())
    }
}

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

    /// SymbolicRegression.jl's `max_features`: the features a tree may use, the widest
    /// subexpression's arity in a template search.
    pub(crate) fn max_features(&self, n_features: usize) -> usize {
        self.template
            .as_ref()
            .map_or(n_features, TemplateStructure::max_features)
    }

    /// The features tree `k` may use: subexpression `k`'s arity in a template search.
    pub(crate) fn tree_features(&self, k: usize, n_features: usize) -> usize {
        self.template.as_ref().map_or(n_features, |t| t.num_features()[k])
    }

    /// Checks, before a search starts, that the loss can score `dataset` and that the template
    /// reads its features: one name per feature, valid parameter indices on every row, and
    /// operators that `D` can differentiate. Templates take no units, as in SymbolicRegression.jl.
    pub fn check_dataset(&self, dataset: &Dataset<T>) -> Result<(), String> {
        self.loss.check_weights(dataset.weights_slice().is_some())?;
        for (mutation, weight) in self.resolved_mutations() {
            let BuiltinMutation::Backsolve(backsolve) = mutation else {
                continue;
            };
            if weight > 0.0 && backsolve.max_library_size < 1 + dataset.n_features {
                return Err(format!(
                    "backsolve's max_library_size must be at least 1 + the number of features ({}), got {}",
                    1 + dataset.n_features,
                    backsolve.max_library_size
                ));
            }
        }
        let Some(template) = &self.template else {
            return Ok(());
        };
        if dataset.units.is_some() {
            return Err("template expressions do not support units".to_string());
        }
        let names = template.variable_names();
        if names.len() != dataset.n_features {
            return Err(format!(
                "the template reads {} features ({}), but the dataset has {}",
                names.len(),
                names.join(", "),
                dataset.n_features
            ));
        }
        template.check_indices(dataset.x.view())?;
        let ops: Vec<_> = self.operators.ops_by_arity.iter().flatten().copied().collect();
        template
            .check_operators(&self.operators.set, &ops)
            .map_err(|e| e.to_string())
    }
}
