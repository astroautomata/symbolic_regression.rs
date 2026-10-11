import React, { useMemo, useState } from "react";
import { useSessionStore } from "../../state/sessionStore";
import type { LossScale, WasmGuess, WasmOpInfo, WasmTemplateOptions } from "../../types/srTypes";

/** Built-in losses, written as PySR's `elementwise_loss` takes them. Any definition such as
 * `f(x, y) = abs(x - y)^1.5` also works. */
const BUILTIN_LOSSES = [
  "L2DistLoss()",
  "L1DistLoss()",
  "LPDistLoss{3}()",
  "HuberLoss(1.0)",
  "L1EpsilonInsLoss(0.1)",
  "L2EpsilonInsLoss(0.1)",
  "LogitDistLoss()",
  "QuantileLoss(0.5)",
  "LogCoshLoss()",
  "PeriodicLoss(1.0)",
  "ZeroOneLoss()",
  "PerceptronLoss()",
  "LogitMarginLoss()",
  "L1HingeLoss()",
  "L2HingeLoss()",
  "SmoothedL1HingeLoss(1.0)",
  "ModifiedHuberLoss()",
  "L2MarginLoss()",
  "ExpLoss()",
  "SigmoidLoss()",
  "DWDMarginLoss(1.0)"
];

function groupByArity(ops: WasmOpInfo[]): Record<number, WasmOpInfo[]> {
  const out: Record<number, WasmOpInfo[]> = {};
  for (const op of ops) {
    (out[op.arity] ??= []).push(op);
  }
  for (const k of Object.keys(out)) out[Number(k)].sort((a, b) => a.name.localeCompare(b.name));
  return out;
}

function splitByName(ops: WasmOpInfo[], basicNames: string[]): { basic: WasmOpInfo[]; more: WasmOpInfo[] } {
  const basicSet = new Set(basicNames);
  const basic: WasmOpInfo[] = [];
  const more: WasmOpInfo[] = [];
  for (const op of ops) {
    if (basicSet.has(op.name)) basic.push(op);
    else more.push(op);
  }
  return { basic, more };
}

function OperatorsGroup(props: {
  title: string;
  arity: 1 | 2;
  ops: WasmOpInfo[];
  selected: string[];
  basicNames: string[];
  toggleOp: (arity: 1 | 2 | 3, name: string) => void;
}): React.ReactElement {
  const { basic, more } = splitByName(props.ops, props.basicNames);
  const renderList = (list: WasmOpInfo[]) => (
    <div className="checkboxGrid">
      {list.map((op) => {
        const checked = props.selected.includes(op.name);
        const label = op.display === op.name ? op.name : `${op.name} (${op.display})`;
        return (
          <label key={`${op.arity}:${op.name}`} className="checkbox">
            <input type="checkbox" checked={checked} onChange={() => props.toggleOp(props.arity, op.name)} />
            <span className="mono">{label}</span>
          </label>
        );
      })}
    </div>
  );

  return (
    <div className="opGroup">
      <div className="subTitle">
        {props.title} ({props.selected.length} selected)
      </div>

      {basic.length > 0 ? renderList(basic) : <div className="muted">No basic operators available.</div>}

      {more.length > 0 && (
        <details className="details">
          <summary>More {props.title.toLowerCase()} operators</summary>
          {renderList(more)}
        </details>
      )}
    </div>
  );
}

function splitList(text: string): string[] {
  return text
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s !== "");
}

/** Comma-separated subexpression arities; blank infers them. A non-integer entry stays NaN, so the
 * search setup reports it. */
function parseArities(text: string): number[] | null {
  const parts = splitList(text);
  return parts.length === 0 ? null : parts.map((s) => (/^\d+$/.test(s) ? Number(s) : NaN));
}

function TemplateCard(props: {
  template: WasmTemplateOptions | null;
  setTemplate: (template: WasmTemplateOptions | null) => void;
}): React.ReactElement {
  const { template, setTemplate } = props;
  // The list fields keep their raw text, so typing "f, " does not drop the trailing comma.
  const [expressionsText, setExpressionsText] = useState(template?.expressions.join(", ") ?? "");
  const [aritiesText, setAritiesText] = useState(template?.num_features?.join(", ") ?? "");
  const patch = (p: Partial<WasmTemplateOptions>) => template && setTemplate({ ...template, ...p });
  const parameters = template?.parameters ?? [];
  const aritiesValid = (parseArities(aritiesText) ?? []).every(Number.isInteger);

  return (
    <div className="card">
      <div className="cardTitle">Template expression</div>
      <label className="checkbox">
        <input
          type="checkbox"
          checked={template !== null}
          onChange={(e) =>
            setTemplate(
              e.target.checked
                ? {
                    combine: "",
                    expressions: splitList(expressionsText),
                    num_features: parseArities(aritiesText),
                    variable_names: null,
                    parameters: []
                  }
                : null
            )
          }
        />
        search a template expression
      </label>
      {template && (
        <>
          <div className="row">
            <label className="field">
              <div className="label">combine</div>
              <input
                type="text"
                placeholder="f(x1, x2) + g(x3) * p[x4]"
                value={template.combine}
                onChange={(e) => patch({ combine: e.target.value })}
              />
            </label>
            <label className="field">
              <div className="label">expressions</div>
              <input
                type="text"
                placeholder="f, g"
                value={expressionsText}
                onChange={(e) => {
                  setExpressionsText(e.target.value);
                  patch({ expressions: splitList(e.target.value) });
                }}
              />
            </label>
            <label className="field">
              <div className="label">arities</div>
              <input
                type="text"
                placeholder="blank infers them, or 2, 1"
                value={aritiesText}
                onChange={(e) => {
                  setAritiesText(e.target.value);
                  patch({ num_features: parseArities(e.target.value) });
                }}
              />
            </label>
          </div>
          {!aritiesValid && <div className="errorText">arities must be comma-separated whole numbers</div>}
          <div className="subTitle">Parameters</div>
          {parameters.map(([name, size], i) => (
            <div className="row" key={i}>
              <label className="field">
                <div className="label">parameter name</div>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => patch({ parameters: parameters.map<[string, number]>((p, j) => (j === i ? [e.target.value, size] : p)) })}
                />
              </label>
              <label className="field">
                <div className="label">parameter size</div>
                <input
                  type="number"
                  min={1}
                  value={size}
                  onChange={(e) => patch({ parameters: parameters.map<[string, number]>((p, j) => (j === i ? [name, Number(e.target.value)] : p)) })}
                />
              </label>
              <button type="button" onClick={() => patch({ parameters: parameters.filter((_, j) => j !== i) })}>
                remove
              </button>
            </div>
          ))}
          <button type="button" onClick={() => patch({ parameters: [...parameters, ["p", 1]] })}>
            add parameter
          </button>
        </>
      )}
    </div>
  );
}

/** A template guess names each part as template results print, `f = #1 * #2; p = [5.0, 10.0]`;
 * any other line is a plain equation. */
function parseGuess(line: string): WasmGuess {
  if (!/^\s*[A-Za-z_]\w*\s*=(?!=)/.test(line)) return line.trim();
  return line
    .split(";")
    .filter((part) => part.trim() !== "")
    .map((part) => {
      const [name, ...rest] = part.split("=");
      const value = rest.join("=").trim();
      return [name.trim(), /^\[.*\]$/.test(value) ? splitList(value.slice(1, -1)).map(Number) : value];
    });
}

function formatGuess(guess: WasmGuess): string {
  if (typeof guess === "string") return guess;
  return guess.map(([name, value]) => `${name} = ${typeof value === "string" ? value : `[${value.join(", ")}]`}`).join("; ");
}

function GuessesCard(props: { guesses: WasmGuess[]; setGuesses: (guesses: WasmGuess[]) => void }): React.ReactElement {
  const [text, setText] = useState(props.guesses.map(formatGuess).join("\n"));
  return (
    <div className="card">
      <div className="cardTitle">Guesses</div>
      <label className="field">
        <div className="label">guesses</div>
        <textarea
          className="textarea"
          rows={3}
          placeholder={"one equation per line, such as 2.0*cos(x0) + x1*x1\na template guess: f = #1 * #2; p = [5.0, 10.0]"}
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            props.setGuesses(
              e.target.value
                .split("\n")
                .filter((line) => line.trim() !== "")
                .map(parseGuess)
            );
          }}
        />
      </label>
    </div>
  );
}

export function ModelingTask(): React.ReactElement {
  const registry = useSessionStore((s) => s.operatorRegistry);
  const options = useSessionStore((s) => s.options);
  const runtime = useSessionStore((s) => s.runtime);
  const setOptionsPatch = useSessionStore((s) => s.setOptionsPatch);

  const unaryOps = useSessionStore((s) => s.unaryOps);
  const binaryOps = useSessionStore((s) => s.binaryOps);
  const ternaryOps = useSessionStore((s) => s.ternaryOps);
  const toggleOp = useSessionStore((s) => s.toggleOp);

  const grouped = useMemo(() => groupByArity(registry), [registry]);

  if (!options) {
    return (
      <div className="pane">
        <div className="card">
          {runtime.status === "error" && runtime.error ? (
            <div className="error">{runtime.error}</div>
          ) : (
            <div className="muted">Loading defaults…</div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="pane">
      <div className="card">
        <div className="cardTitle">Operators</div>

        <div className="opGroups">
          <OperatorsGroup
            title="Unary"
            arity={1}
            ops={grouped[1] ?? []}
            selected={unaryOps}
            basicNames={["abs", "sin", "cos", "sqrt", "log", "exp"]}
            toggleOp={toggleOp}
          />
          <OperatorsGroup
            title="Binary"
            arity={2}
            ops={grouped[2] ?? []}
            selected={binaryOps}
            basicNames={["add", "sub", "mul", "div", "pow"]}
            toggleOp={toggleOp}
          />
          {ternaryOps.length > 0 && (
            <div className="muted">Note: Ternary operators are hidden in the UI (currently {ternaryOps.length} selected).</div>
          )}
        </div>
      </div>

      <div className="card">
        <div className="cardTitle">Loss</div>
        <div className="row">
          <label className="field">
            <div className="label">loss</div>
            <input type="text" list="builtin-losses" value={options.loss} onChange={(e) => setOptionsPatch({ loss: e.target.value })} />
            <datalist id="builtin-losses">
              {BUILTIN_LOSSES.map((loss) => (
                <option key={loss} value={loss} />
              ))}
            </datalist>
          </label>
          <label className="field">
            <div className="label">loss_scale</div>
            <select value={options.loss_scale} onChange={(e) => setOptionsPatch({ loss_scale: e.target.value as LossScale })}>
              <option value="log">log</option>
              <option value="linear">linear</option>
            </select>
          </label>
          <label className="field">
            <div className="label">early_stop_condition</div>
            <input
              type="text"
              placeholder="1e-6 or f(loss, complexity) = loss < 1e-6 && complexity < 10"
              value={options.early_stop_condition ?? ""}
              onChange={(e) => setOptionsPatch({ early_stop_condition: e.target.value.trim() === "" ? null : e.target.value })}
            />
          </label>
        </div>
      </div>

      <TemplateCard template={options.template ?? null} setTemplate={(template) => setOptionsPatch({ template })} />

      <GuessesCard guesses={options.guesses ?? []} setGuesses={(guesses) => setOptionsPatch({ guesses })} />

      <div className="card">
        <div className="cardTitle">Hyperparameters</div>
        <div className="muted">All fields map 1:1 to `symbolic_regression::Options` (plus a few web-only ones).</div>

        <div className="row">
          <label className="field">
            <div className="label">maxsize</div>
            <input type="number" min={1} value={options.maxsize} onChange={(e) => setOptionsPatch({ maxsize: Number(e.target.value) })} />
          </label>
          <label className="field">
            <div className="label">seed</div>
            <input type="number" min={0} value={options.seed} onChange={(e) => setOptionsPatch({ seed: Number(e.target.value) })} />
          </label>
        </div>

        <details className="details">
          <summary>Advanced hyperparameters</summary>
          <div className="grid2">
            <section className="section">
              <div className="subTitle">Search Budget</div>
              <label className="field">
                <div className="label">populations</div>
                <input
                  type="number"
                  min={1}
                  value={options.populations}
                  onChange={(e) => setOptionsPatch({ populations: Number(e.target.value) })}
                  data-testid="opt-populations"
                />
              </label>
              <label className="field">
                <div className="label">population_size</div>
                <input
                  type="number"
                  min={1}
                  value={options.population_size}
                  onChange={(e) => setOptionsPatch({ population_size: Number(e.target.value) })}
                  data-testid="opt-population-size"
                />
              </label>
              <label className="field">
                <div className="label">ncycles_per_iteration</div>
                <input
                  type="number"
                  min={1}
                  value={options.ncycles_per_iteration}
                  onChange={(e) => setOptionsPatch({ ncycles_per_iteration: Number(e.target.value) })}
                  data-testid="opt-ncycles"
                />
              </label>
            </section>

            <section className="section">
              <div className="subTitle">Constraints</div>
              <label className="field">
                <div className="label">maxdepth</div>
                <input type="number" min={1} value={options.maxdepth} onChange={(e) => setOptionsPatch({ maxdepth: Number(e.target.value) })} />
              </label>
              <label className="field">
                <div className="label">warmup_maxsize_by</div>
                <input type="number" step={0.05} value={options.warmup_maxsize_by} onChange={(e) => setOptionsPatch({ warmup_maxsize_by: Number(e.target.value) })} />
              </label>
              <label className="field">
                <div className="label">topn</div>
                <input type="number" min={1} value={options.topn} onChange={(e) => setOptionsPatch({ topn: Number(e.target.value) })} />
              </label>
              <label className="field">
                <div className="label">dimensional_constraint_penalty</div>
                <input
                  type="number"
                  value={options.dimensional_constraint_penalty}
                  onChange={(e) => setOptionsPatch({ dimensional_constraint_penalty: Number(e.target.value) })}
                />
              </label>
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={options.dimensionless_constants_only}
                  onChange={(e) => setOptionsPatch({ dimensionless_constants_only: e.target.checked })}
                />
                dimensionless_constants_only
              </label>
            </section>

            <section className="section">
              <div className="subTitle">Selection</div>
              <label className="field">
                <div className="label">tournament_selection_n</div>
                <input type="number" min={1} value={options.tournament_selection_n} onChange={(e) => setOptionsPatch({ tournament_selection_n: Number(e.target.value) })} />
              </label>
              <label className="field">
                <div className="label">tournament_selection_p</div>
                <input type="number" min={0} max={1} step={0.001} value={options.tournament_selection_p} onChange={(e) => setOptionsPatch({ tournament_selection_p: Number(e.target.value) })} />
              </label>
              <label className="checkbox">
                <input type="checkbox" checked={options.use_frequency} onChange={(e) => setOptionsPatch({ use_frequency: e.target.checked })} />
                use_frequency
              </label>
              <label className="checkbox">
                <input type="checkbox" checked={options.use_frequency_in_tournament} onChange={(e) => setOptionsPatch({ use_frequency_in_tournament: e.target.checked })} />
                use_frequency_in_tournament
              </label>
            </section>

            <section className="section">
              <div className="subTitle">Parsimony / Scaling</div>
              <label className="field">
                <div className="label">parsimony</div>
                <input type="number" step={0.001} value={options.parsimony} onChange={(e) => setOptionsPatch({ parsimony: Number(e.target.value) })} />
              </label>
              <label className="field">
                <div className="label">adaptive_parsimony_scaling</div>
                <input type="number" step={0.1} value={options.adaptive_parsimony_scaling} onChange={(e) => setOptionsPatch({ adaptive_parsimony_scaling: Number(e.target.value) })} />
              </label>
              <label className="checkbox">
                <input type="checkbox" checked={options.use_baseline} onChange={(e) => setOptionsPatch({ use_baseline: e.target.checked })} />
                use_baseline
              </label>
              <label className="checkbox">
                <input type="checkbox" checked={options.annealing} onChange={(e) => setOptionsPatch({ annealing: e.target.checked })} />
                annealing
              </label>
            </section>

            <section className="section">
              <div className="subTitle">Mutation / Crossover</div>
              <label className="field">
                <div className="label">crossover_probability</div>
                <input type="number" step={0.001} value={options.crossover_probability} onChange={(e) => setOptionsPatch({ crossover_probability: Number(e.target.value) })} />
              </label>
              <label className="field">
                <div className="label">perturbation_factor</div>
                <input type="number" step={0.001} value={options.perturbation_factor} onChange={(e) => setOptionsPatch({ perturbation_factor: Number(e.target.value) })} />
              </label>
              <label className="field">
                <div className="label">probability_negate_constant</div>
                <input type="number" step={0.001} value={options.probability_negate_constant} onChange={(e) => setOptionsPatch({ probability_negate_constant: Number(e.target.value) })} />
              </label>
              <label className="checkbox">
                <input type="checkbox" checked={options.skip_mutation_failures} onChange={(e) => setOptionsPatch({ skip_mutation_failures: e.target.checked })} />
                skip_mutation_failures
              </label>
              <label className="checkbox">
                <input type="checkbox" checked={options.should_simplify} onChange={(e) => setOptionsPatch({ should_simplify: e.target.checked })} />
                should_simplify
              </label>
            </section>

            <section className="section">
              <div className="subTitle">Optimization</div>
              <label className="checkbox">
                <input type="checkbox" checked={options.should_optimize_constants} onChange={(e) => setOptionsPatch({ should_optimize_constants: e.target.checked })} />
                should_optimize_constants
              </label>
              <label className="field">
                <div className="label">optimizer_probability</div>
                <input type="number" step={0.001} value={options.optimizer_probability} onChange={(e) => setOptionsPatch({ optimizer_probability: Number(e.target.value) })} />
              </label>
              <label className="field">
                <div className="label">optimizer_iterations</div>
                <input type="number" min={0} value={options.optimizer_iterations} onChange={(e) => setOptionsPatch({ optimizer_iterations: Number(e.target.value) })} />
              </label>
              <label className="field">
                <div className="label">optimizer_nrestarts</div>
                <input type="number" min={0} value={options.optimizer_nrestarts} onChange={(e) => setOptionsPatch({ optimizer_nrestarts: Number(e.target.value) })} />
              </label>
              <label className="field">
                <div className="label">optimizer_f_calls_limit</div>
                <input type="number" min={0} value={options.optimizer_f_calls_limit} onChange={(e) => setOptionsPatch({ optimizer_f_calls_limit: Number(e.target.value) })} />
              </label>
            </section>

            <section className="section">
              <div className="subTitle">Migration / Replacement</div>
              <label className="checkbox">
                <input type="checkbox" checked={options.migration} onChange={(e) => setOptionsPatch({ migration: e.target.checked })} />
                migration
              </label>
              <label className="checkbox">
                <input type="checkbox" checked={options.hof_migration} onChange={(e) => setOptionsPatch({ hof_migration: e.target.checked })} />
                hof_migration
              </label>
              <label className="field">
                <div className="label">fraction_replaced</div>
                <input type="number" step={0.0001} value={options.fraction_replaced} onChange={(e) => setOptionsPatch({ fraction_replaced: Number(e.target.value) })} />
              </label>
              <label className="field">
                <div className="label">fraction_replaced_hof</div>
                <input type="number" step={0.0001} value={options.fraction_replaced_hof} onChange={(e) => setOptionsPatch({ fraction_replaced_hof: Number(e.target.value) })} />
              </label>
              <label className="field">
                <div className="label">fraction_replaced_guesses</div>
                <input
                  type="number"
                  step={0.0001}
                  value={options.fraction_replaced_guesses}
                  onChange={(e) => setOptionsPatch({ fraction_replaced_guesses: Number(e.target.value) })}
                />
              </label>
            </section>

            <section className="section">
              <div className="subTitle">Misc</div>
              <label className="field">
                <div className="label">alpha</div>
                <input type="number" step={0.001} value={options.alpha} onChange={(e) => setOptionsPatch({ alpha: Number(e.target.value) })} />
              </label>
            </section>

            <section className="section fullWidth">
              <div className="subTitle">Mutation weights</div>
              <div className="mutationWeightsGrid">
                {Object.entries(options.mutation_weights ?? {}).map(([k, v]) => (
                  <label key={k} className="field">
                    <div className="label">{k}</div>
                    <input
                      type="number"
                      step={0.0001}
                      value={v}
                      onChange={(e) =>
                        setOptionsPatch({
                          mutation_weights: { ...options.mutation_weights, [k]: Number(e.target.value) } as any
                        })
                      }
                    />
                  </label>
                ))}
              </div>
            </section>
          </div>
        </details>
      </div>
    </div>
  );
}
