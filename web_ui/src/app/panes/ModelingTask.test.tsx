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
