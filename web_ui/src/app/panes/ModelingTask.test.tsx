import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { ModelingTask } from "./ModelingTask";
import { useSessionStore } from "../../state/sessionStore";
import { default_search_options } from "../../pkg/symbolic_regression_wasm.js";
import type { WasmSearchOptions } from "../../types/srTypes";

const template = () => useSessionStore.getState().options?.template;
const type = (label: string, value: string) => fireEvent.change(screen.getByLabelText(label), { target: { value } });

describe("ModelingTask template expression", () => {
  beforeEach(async () => {
    const options = (await default_search_options()) as WasmSearchOptions;
    useSessionStore.setState({ options, operatorRegistry: [], unaryOps: [], binaryOps: [], ternaryOps: [] });
  });

  it("updates the dimensional penalty and toggles dimensionless constants", () => {
    render(<ModelingTask />);
    type("dimensional_constraint_penalty", "250");
    expect(useSessionStore.getState().options?.dimensional_constraint_penalty).toBe(250);
    const toggle = screen.getByLabelText("dimensionless_constants_only");
    expect(toggle).not.toBeChecked();
    fireEvent.click(toggle);
    expect(useSessionStore.getState().options?.dimensionless_constants_only).toBe(true);
    fireEvent.click(toggle);
    expect(useSessionStore.getState().options?.dimensionless_constants_only).toBe(false);
  });
  it("is off by default and toggles an empty template on and off", () => {
    render(<ModelingTask />);
    const toggle = screen.getByLabelText("search a template expression");
    expect(toggle).not.toBeChecked();
    expect(template()).toBeNull();
    expect(screen.queryByLabelText("combine")).toBeNull();

    fireEvent.click(toggle);
    expect(template()).toEqual({ combine: "", expressions: [], num_features: null, variable_names: null, parameters: [] });

    fireEvent.click(toggle);
    expect(template()).toBeNull();
  });

  it("writes combine, expression names, arities, and parameter rows into the options payload", () => {
    render(<ModelingTask />);
    fireEvent.click(screen.getByLabelText("search a template expression"));

    type("combine", "f(x, y) + g(x) * p[c]");
    type("expressions", "f, g,");
    expect(screen.getByLabelText("expressions")).toHaveValue("f, g,");
    expect(template()).toMatchObject({ combine: "f(x, y) + g(x) * p[c]", expressions: ["f", "g"], num_features: null });

    type("arities", "3, 1");
    expect(template()?.num_features).toEqual([3, 1]);
    expect(screen.queryByText("arities must be comma-separated whole numbers")).toBeNull();
    type("arities", "3, one");
    expect(template()?.num_features).toEqual([3, NaN]);
    expect(screen.getByText("arities must be comma-separated whole numbers")).toBeInTheDocument();
    type("arities", " ");
    expect(template()?.num_features).toBeNull();

    fireEvent.click(screen.getByText("add parameter"));
    fireEvent.click(screen.getByText("add parameter"));
    fireEvent.change(screen.getAllByLabelText("parameter name")[0], { target: { value: "q" } });
    fireEvent.change(screen.getAllByLabelText("parameter size")[0], { target: { value: "2" } });
    fireEvent.change(screen.getAllByLabelText("parameter size")[1], { target: { value: "3" } });
    expect(template()?.parameters).toEqual([
      ["q", 2],
      ["p", 3]
    ]);

    fireEvent.click(screen.getAllByText("remove")[0]);
    expect(template()?.parameters).toEqual([["p", 3]]);
    expect(template()?.variable_names).toBeNull();
  });
});

describe("ModelingTask guesses", () => {
  beforeEach(async () => {
    const options = (await default_search_options()) as WasmSearchOptions;
    useSessionStore.setState({ options, operatorRegistry: [], unaryOps: [], binaryOps: [], ternaryOps: [] });
  });

  const guesses = () => useSessionStore.getState().options?.guesses;

  it("writes one guess per line, reading template guesses as the results print them", () => {
    const { unmount } = render(<ModelingTask />);
    expect(guesses()).toEqual([]);

    type("guesses", "2.0*cos(x0) + x1*x1\n\n  x0 == 1.0 \nf = #1 * #2; g = cos(#1) ; p = [5.0, -1e-3];\n");
    expect(guesses()).toEqual([
      "2.0*cos(x0) + x1*x1",
      "x0 == 1.0",
      [
        ["f", "#1 * #2"],
        ["g", "cos(#1)"],
        ["p", [5, -0.001]]
      ]
    ]);

    unmount();
    render(<ModelingTask />);
    expect(screen.getByLabelText("guesses")).toHaveValue("2.0*cos(x0) + x1*x1\nx0 == 1.0\nf = #1 * #2; g = cos(#1); p = [5, -0.001]");

    type("guesses", " ");
    expect(guesses()).toEqual([]);
  });

  it("updates fraction_replaced_guesses", () => {
    render(<ModelingTask />);
    type("fraction_replaced_guesses", "0.05");
    expect(useSessionStore.getState().options?.fraction_replaced_guesses).toBe(0.05);
  });
});

describe("ModelingTask search options", () => {
  let defaults: WasmSearchOptions;
  beforeEach(async () => {
    defaults = (await default_search_options()) as WasmSearchOptions;
    useSessionStore.setState({ options: defaults, operatorRegistry: [], unaryOps: [], binaryOps: [], ternaryOps: [] });
  });

  const options = () => useSessionStore.getState().options!;
  // A label wrapping a <select> also contains the option texts.
  const choose = (label: RegExp, value: string) => fireEvent.change(screen.getByLabelText(label), { target: { value } });

  it("shows the SymbolicRegression.jl defaults", () => {
    render(<ModelingTask />);
    expect(screen.getByLabelText("backsolve")).toHaveValue(0);
    expect(screen.getByLabelText("max_library_size")).toHaveValue(500);
    expect(screen.getByLabelText("max_terms")).toHaveValue(8);
    expect(screen.getByLabelText("min_improvement")).toHaveValue(0.001);
    expect(screen.getByLabelText("node_attempts")).toHaveValue(8);
    expect(screen.getByLabelText("adaptive_mutation_weights")).toBeChecked();
    expect(screen.getByLabelText("smoothing")).toHaveValue(0.02);
    expect(screen.getByLabelText("floor")).toHaveValue(0.05);
    expect(screen.getByLabelText("adaptation_strength")).toHaveValue(0.5);
    expect(screen.getByLabelText(/^reward/)).toHaveValue("cost");
    expect(screen.getByLabelText("mutation_burst")).not.toBeChecked();
    expect(screen.queryByLabelText("retry_attempts")).toBeNull();
    expect(screen.getByLabelText(/^batching/)).toHaveValue("auto");
    expect(screen.getByLabelText("batch_size")).toHaveValue(null);
    expect(screen.getByLabelText(/^optimizer_algorithm/)).toHaveValue("BFGS");
    expect(screen.getByLabelText("timeout_in_seconds")).toHaveValue(0);
    expect(screen.getByLabelText("max_evals")).toHaveValue(0);
    expect(screen.getByLabelText("alpha")).toHaveValue(3.17);
  });

  it("patches the backsolve weight and backsolve parameters", () => {
    render(<ModelingTask />);
    type("backsolve", "0.5");
    expect(options().mutation_weights).toEqual({ ...defaults.mutation_weights, backsolve: 0.5 });
    type("max_terms", "3");
    type("min_improvement", "0.01");
    expect(options().backsolve).toEqual({ max_library_size: 500, max_terms: 3, min_improvement: 0.01, node_attempts: 8 });
  });

  it("edits adaptive mutation weights and switches them off and back to their defaults", () => {
    render(<ModelingTask />);
    type("smoothing", "0.1");
    choose(/^reward/, "loss");
    expect(options().adaptive_mutation_weights).toEqual({ smoothing: 0.1, floor: 0.05, reward: "loss", adaptation_strength: 0.5 });

    fireEvent.click(screen.getByLabelText("adaptive_mutation_weights"));
    expect(options().adaptive_mutation_weights).toBeNull();
    expect(screen.queryByLabelText("smoothing")).toBeNull();

    fireEvent.click(screen.getByLabelText("adaptive_mutation_weights"));
    expect(options().adaptive_mutation_weights).toEqual(defaults.adaptive_mutation_weights);
  });

  it("switches mutation bursts on with their defaults and edits them", () => {
    render(<ModelingTask />);
    fireEvent.click(screen.getByLabelText("mutation_burst"));
    expect(options().mutation_burst).toEqual({ retry_attempts: 4, compound_probability: 0.25, compound_max_steps: 2 });

    type("retry_attempts", "6");
    type("compound_probability", "0.5");
    type("compound_max_steps", "3");
    expect(options().mutation_burst).toEqual({ retry_attempts: 6, compound_probability: 0.5, compound_max_steps: 3 });

    fireEvent.click(screen.getByLabelText("mutation_burst"));
    expect(options().mutation_burst).toBeNull();
  });

  it("patches batching, the optimizer algorithm, and the stopping limits", () => {
    render(<ModelingTask />);
    choose(/^batching/, "on");
    type("batch_size", "64");
    expect([options().batching, options().batch_size]).toEqual(["on", 64]);
    type("batch_size", "");
    expect(options().batch_size).toBeNull();

    choose(/^optimizer_algorithm/, "NelderMead");
    expect(options().optimizer_algorithm).toBe("NelderMead");

    type("timeout_in_seconds", "30");
    type("max_evals", "5000");
    type("alpha", "1.5");
    expect([options().timeout_in_seconds, options().max_evals, options().alpha]).toEqual([30, 5000, 1.5]);
  });
});
