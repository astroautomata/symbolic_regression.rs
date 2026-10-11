export type WasmOpInfo = {
  arity: number;
  name: string;
  display: string;
  infix: string | null;
  commutative: boolean;
  associative: boolean;
  complexity: number;
};

export type WasmMutationWeights = {
  mutate_constant: number;
  mutate_operator: number;
  mutate_feature: number;
  swap_operands: number;
  rotate_tree: number;
  add_node: number;
  insert_node: number;
  delete_node: number;
  simplify: number;
  randomize: number;
  do_nothing: number;
  optimize: number;
  form_connection: number;
  break_connection: number;
};

export type LossScale = "log" | "linear";

/** A template expression, as PySR's `TemplateExpressionSpec`. */
export type WasmTemplateOptions = {
  /** The formula combining the subexpressions, such as "f(x1, x2) + p[x3]". */
  combine: string;
  /** Subexpression names, such as ["f", "g"]. */
  expressions: string[];
  /** Each subexpression's number of arguments; null infers them from `combine`. */
  num_features: number[] | null;
  /** The names `combine` uses for the features; null uses the selected columns' CSV headers. */
  variable_names: string[] | null;
  /** Parameter vectors as [name, length], in declaration order. */
  parameters: [string, number][];
};

/** A template guess's value for one name: a subexpression over its arguments #1, #2, ..., or a parameter vector. */
export type WasmTemplateGuessValue = string | number[];

/** A guess that seeds the search: an equation, or a template search's [name, value] pairs, as PySR's dict guesses. */
export type WasmGuess = string | [string, WasmTemplateGuessValue][];

export type WasmBuiltinMutation =
  | { Constant: { perturbation_factor: number; probability_negate: number } }
  | "Operator" | "Feature" | "SwapOperands" | "RotateTree" | "AddNode"
  | "InsertNode" | "DeleteNode" | "Simplify" | "Randomize" | "DoNothing" | "Optimize";

export type WasmSearchOptions = {
  has_headers: boolean;
  x_columns: number[] | null;
  y_column: number | null;
  weights_column: number | null;
  validation_fraction: number;
  /** One PySR unit string per selected x column, in x_columns order. */
  x_units: string[] | null;
  /** PySR unit string for the target column. */
  y_units: string | null;
  /** PySR `elementwise_loss`: a built-in such as "L1DistLoss()" or a definition such as "f(x, y) = abs(x - y)". */
  loss: string;
  /** PySR `early_stop_condition`: a loss threshold such as "1e-6", or a definition of (loss, complexity). */
  early_stop_condition: string | null;
  /** A template expression; null searches plain expressions. */
  template: WasmTemplateOptions | null;
  /** SymbolicRegression.jl `guesses`: equations that seed the search. */
  guesses: WasmGuess[];

  seed: number | null;
  niterations: number;
  populations: number;
  population_size: number;
  ncycles_per_iteration: number;
  maxsize: number;
  maxdepth: number | null;
  warmup_maxsize_by: number;
  parsimony: number;
  adaptive_parsimony_scaling: number;
  crossover_probability: number;
  perturbation_factor: number;
  probability_negate_constant: number;
  tournament_selection_n: number;
  tournament_selection_p: number;
  alpha: number;
  optimizer_nrestarts: number;
  optimizer_probability: number;
  optimizer_algorithm: "BFGS" | "NelderMead";
  optimizer_g_abstol: number;
  optimizer_x_abstol: number;
  optimizer_f_reltol: number;
  optimizer_g_calls_limit: number;
  parallelism: "serial" | "multithreading";
  batching: "auto" | "on" | "off";
  batch_size: number | null;
  optimizer_iterations: number;
  optimizer_f_calls_limit: number;
  fraction_replaced: number;
  fraction_replaced_hof: number;
  fraction_replaced_guesses: number;
  topn: number;
  print_precision: number;
  loss_scale: LossScale;
  dimensional_constraint_penalty: number;

  use_frequency: boolean;
  use_frequency_in_tournament: boolean;
  skip_mutation_failures: boolean;
  annealing: boolean;
  should_optimize_constants: boolean;
  migration: boolean;
  hof_migration: boolean;
  use_baseline: boolean;
  progress: boolean;
  should_simplify: boolean | null;
  dimensionless_constants_only: boolean;

  mutations: Array<[WasmBuiltinMutation, number]>;
  default_mutations: Array<[WasmBuiltinMutation, number]> | null;
  crossovers: Array<["Subtree", number]>;
  default_crossovers: Array<["Subtree", number]> | null;
  mutation_weights: WasmMutationWeights;
};

export type EquationPoint = {
  id: string;
  complexity: number;
  loss: number;
  cost: number;
  /** The Pareto score, as in SymbolicRegression.jl's hall-of-fame table. */
  score: number;
};

export type EquationSummary = {
  id: string;
  complexity: number;
  loss: number;
  cost: number;
  /** The Pareto score, as in SymbolicRegression.jl's hall-of-fame table. */
  score: number;
  /** Display-precision equation (matches `print_precision`, default 5 significant digits). */
  equation: string;
  /** Full-precision equation, suitable for copy-to-clipboard. */
  equation_full: string;
};

export type SearchSnapshot = {
  total_cycles: number;
  cycles_completed: number;
  total_evals: number;
  /** The equation SymbolicRegression.jl reports as best (`choose_best` under `loss_scale`). */
  best: EquationSummary | null;
  pareto_points: EquationPoint[];
};

export type WasmSplitIndices = {
  train: number[];
  val: number[];
};

export type WasmMetrics = {
  n: number;
  mse: number;
  mae: number;
  rmse: number;
  r2: number;
  corr: number;
  min_abs_err: number;
  max_abs_err: number;
};

export type WasmEvalResult = {
  metrics: WasmMetrics;
  yhat: number[];
};
