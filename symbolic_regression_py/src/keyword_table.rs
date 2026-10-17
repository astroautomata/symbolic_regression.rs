//! Shared option classification for conversion, Python metadata, and source parity.

macro_rules! keyword_table {
    ($consumer:ident $(, $args:ident)*) => {
        $consumer! {
            [$($args),*]
            regular {
                constructor plain to_usize: ncycles_per_iteration, optimizer_nrestarts, optimizer_iterations, optimizer_f_calls_limit, topn, maxsize;
                search plain to_usize: niterations;
                rust plain to_usize: optimizer_g_calls_limit;
                constructor plain to_cost: warmup_maxsize_by, complexity_of_constants;
                constructor plain to_positive: populations, population_size, tournament_selection_n, print_precision;
                constructor plain to_nonnegative: parsimony, adaptive_parsimony_scaling, alpha, perturbation_factor, timeout_in_seconds, dimensional_constraint_penalty;
                rust plain to_nonnegative: optimizer_g_abstol, optimizer_x_abstol, optimizer_f_reltol;
                constructor plain to_probability: crossover_probability, probability_negate_constant, optimizer_probability, fraction_replaced, fraction_replaced_hof, fraction_replaced_guesses;
                constructor plain to_probability32: tournament_selection_p;
                constructor plain to_bool: dimensionless_constants_only, use_frequency, use_frequency_in_tournament, skip_mutation_failures, annealing, should_optimize_constants, migration, hof_migration, deterministic, save_to_file, progress, use_tracing;
                rust plain to_bool: use_baseline;
                constructor plain to_u64: max_evals;
                constructor optional to_positive: maxdepth, terminal_width;
                constructor optional to_usize: batch_size;
                constructor optional to_bool: should_simplify;
                constructor optional to_i32: verbosity;
                constructor optional to_u64: seed;
                constructor optional to_string: output_directory;
                search optional to_string: run_id;
            }
            special {
                constructor defaults;
                constructor operators;
                constructor binary_operators;
                constructor unary_operators;
                rust output_style;
                constructor tracing_file;
                constructor loss_scale;
                constructor optimizer_algorithm;
                constructor optimizer_options;
                constructor batching;
                search parallelism;
                constructor input_stream;
                constructor elementwise_loss;
                constructor early_stop_condition;
                constructor complexity_of_variables;
                constructor complexity_of_operators;
                constructor constraints;
                constructor nested_constraints;
                constructor expression_spec;
                constructor mutation_weights;
                constructor default_mutations;
                constructor mutations;
                constructor default_crossovers;
                constructor crossovers;
                constructor plugins;
                constructor default_plugins;
            }
            rejected {
                constructor loss_function: "it takes a Julia function of the whole expression; use `elementwise_loss`";
                constructor loss_function_expression: "it takes a Julia function of the whole expression; use `elementwise_loss`";
                constructor complexity_mapping: "it takes a Julia function; use the `complexity_of_*` options";
                constructor operator_enum_constructor: "it takes a Julia type";
                constructor popmember_type: "it takes a Julia type";
                search loss_type: "it takes a Julia type";
                constructor turbo: "it selects LoopVectorization.jl";
                constructor bumper: "it selects Bumper.jl";
                constructor autodiff_backend: "it selects a Julia automatic differentiation package";
                constructor define_helper_functions: "it defines Julia functions";
                search logger: "it takes a Julia logger";
                search external_stop: "the Python binding handles cancellation and Ctrl-C";
                search numprocs: "multiprocessing runs Julia worker processes";
                search procs: "multiprocessing runs Julia worker processes";
                search addprocs_function: "multiprocessing runs Julia worker processes";
                search heap_size_hint_in_bytes: "multiprocessing runs Julia worker processes";
                search worker_timeout: "multiprocessing runs Julia worker processes";
                search worker_imports: "multiprocessing runs Julia worker processes";
                search runtests: "it tests Julia worker processes";
                search extra: "it passes arbitrary Julia data to a custom evaluation function";
                search multithreaded: "it is a deprecated alias; use `parallelism`";
                alias return_state: "the search always returns its state";
                constructor deprecated_return_state: "it is deprecated; the search always returns its state";
                constructor npopulations: "it is a deprecated alias; use `populations`";
                constructor npop: "it is a deprecated alias; use `population_size`";
                constructor expression_type: "it is deprecated; use `expression_spec`";
                constructor expression_options: "it is deprecated; use `expression_spec`";
                constructor node_type: "it is deprecated; use `expression_spec`";
                constructor output_file: "it is deprecated; use `output_directory`";
                constructor fast_cycle: "it is deprecated";
                constructor bin_constraints: "it is deprecated; use `constraints`";
                constructor una_constraints: "it is deprecated; use `constraints`";
                alias mutationWeights: "it is a deprecated alias; use `mutation_weights`";
                alias hofMigration: "it is a deprecated alias; use `hof_migration`";
                alias shouldOptimizeConstants: "it is a deprecated alias; use `should_optimize_constants`";
                alias perturbationFactor: "it is a deprecated alias; use `perturbation_factor`";
                alias batchSize: "it is a deprecated alias; use `batch_size`";
                alias crossoverProbability: "it is a deprecated alias; use `crossover_probability`";
                alias warmupMaxsizeBy: "it is a deprecated alias; use `warmup_maxsize_by`";
                alias useFrequency: "it is a deprecated alias; use `use_frequency`";
                alias useFrequencyInTournament: "it is a deprecated alias; use `use_frequency_in_tournament`";
                alias ncyclesperiteration: "it is a deprecated alias; use `ncycles_per_iteration`";
                alias fractionReplaced: "it is a deprecated alias; use `fraction_replaced`";
                alias fractionReplacedHof: "it is a deprecated alias; use `fraction_replaced_hof`";
                alias probNegate: "it is a deprecated alias; use `probability_negate_constant`";
                alias optimize_probability: "it is a deprecated alias; use `optimizer_probability`";
                alias probPickFirst: "it is a deprecated alias; use `tournament_selection_p`";
                alias earlyStopCondition: "it is a deprecated alias; use `early_stop_condition`";
                alias stateReturn: "it is deprecated; the search always returns its state";
                alias enable_autodiff: "it is deprecated; automatic differentiation requires a Julia package";
                alias ns: "it is a deprecated alias; use `tournament_selection_n`";
                alias loss: "it is a deprecated alias; use `elementwise_loss`";
                alias use_recorder: "it is a deprecated alias; use `use_tracing`";
                alias recorder_file: "it is a deprecated alias; use `tracing_file`";
            }
        }
    };
}

pub(crate) use keyword_table;

macro_rules! metadata {
    ([]
        regular { $($origin:ident $mode:ident $convert:ident: $($field:ident),+;)* }
        special { $($special_origin:ident $special:ident;)* }
        rejected { $($rejected_origin:ident $rejected:ident: $reason:literal;)* }
    ) => {
        pub(crate) const SUPPORTED_OPTIONS: &[&str] = &[$($(stringify!($field),)+)* $(stringify!($special),)*];
        pub(crate) const JULIA_ONLY: &[(&str, &str)] = &[$((stringify!($rejected), $reason),)*];
    };
}
keyword_table!(metadata);
