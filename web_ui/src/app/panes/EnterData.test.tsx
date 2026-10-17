import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { EnterData } from "./EnterData";
import { useSessionStore } from "../../state/sessionStore";
import { vi } from "vitest";
import userEvent from "@testing-library/user-event";

function minimalOptions() {
  return {
    has_headers: true,
    x_columns: [0],
    y_column: 1,
    weights_column: null,
    validation_fraction: 0,
    x_units: null,
    y_units: null,
    loss: "L2DistLoss()",
    early_stop_condition: null,
    template: null,

    seed: 0,
    niterations: 100,
    populations: 2,
    population_size: 16,
    ncycles_per_iteration: 10,
    maxsize: 30,
    maxdepth: 30,
    warmup_maxsize_by: 0,
    parsimony: 0,
    adaptive_parsimony_scaling: 1040,
    crossover_probability: 0.0259,
    perturbation_factor: 0.129,
    probability_negate_constant: 0.00743,
    tournament_selection_n: 15,
    tournament_selection_p: 0.982,
    alpha: 3.17,
    optimizer_nrestarts: 2,
    optimizer_probability: 0.14,
    optimizer_iterations: 8,
    optimizer_f_calls_limit: 10_000,
    fraction_replaced: 0.00036,
    fraction_replaced_hof: 0.0614,
    topn: 12,
    print_precision: 5,
    loss_scale: "log",
    dimensional_constraint_penalty: 1000,

    use_frequency: true,
    use_frequency_in_tournament: true,
    skip_mutation_failures: true,
    annealing: true,
    should_optimize_constants: true,
    migration: true,
    hof_migration: true,
    use_baseline: true,
    progress: false,
    should_simplify: false,
    dimensionless_constants_only: false,

    mutation_weights: {
      mutate_constant: 0.0346,
      mutate_operator: 0.293,
      mutate_feature: 0.1,
      swap_operands: 0.198,
      rotate_tree: 4.26,
      add_node: 2.47,
      insert_node: 0.0112,
      delete_node: 0.87,
      simplify: 0.00209,
      randomize: 0.000502,
      do_nothing: 0.273,
      optimize: 0.0,
      form_connection: 0.5,
      break_connection: 0.1
    }
  };
}

describe("EnterData", () => {
  beforeEach(() => {
    useSessionStore.setState({
      options: minimalOptions() as any,
      parsed: null,
      runtime: {
        status: "idle",
        error: null,
        split: null,
        snapshot: null,
        front: [],
        selectedId: null,
        evalByKey: {}
      }
    });
  });

  it("stores comma-separated x units and y units, and clears blank fields", async () => {
    useSessionStore.setState({
      parsed: { headers: ["a", "b", "y"], rows: [[1, 2, 3]], nRows: 1, nCols: 3 } as any
    });
    render(<EnterData />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("x units"), "m, s");
    await user.type(screen.getByLabelText("y units"), "m/s");
    expect(useSessionStore.getState().options).toMatchObject({ x_units: ["m", "s"], y_units: "m/s" });
    fireEvent.change(screen.getByLabelText("x units"), { target: { value: " " } });
    fireEvent.change(screen.getByLabelText("y units"), { target: { value: " " } });
    expect(useSessionStore.getState().options).toMatchObject({ x_units: null, y_units: null });
  });
  it("accepts drag/drop CSV and updates store csvText", async () => {
    const parseSpy = vi.fn();
    useSessionStore.setState({ parseCsv: parseSpy as any });

    render(<EnterData />);

    const zoneTitle = screen.getByText("Drag & drop a CSV here");
    const dropzone = zoneTitle.closest(".dropzone");
    expect(dropzone).toBeTruthy();

    const csv = "a,b\n1,2\n3,4\n";
    const file = new File([csv], "data.csv", { type: "text/csv" });

    fireEvent.drop(dropzone as Element, {
      dataTransfer: { files: [file] }
    });

    await waitFor(() => {
      expect(useSessionStore.getState().csvText).toContain("a,b");
    });
    await waitFor(() => {
      expect(parseSpy).toHaveBeenCalled();
    });
  });
});
