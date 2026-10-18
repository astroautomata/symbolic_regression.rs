import React from "react";
import { render, screen } from "@testing-library/react";
import { SolutionsTableCard } from "./SolutionsTableCard";

const front = [{ id: "1", complexity: 3, loss: 0.25, cost: 0.25, score: 1.5, equation: "x", equation_full: "x" }];

describe("SolutionsTableCard", () => {
  it("shows the score column for the log loss scale", () => {
    render(<SolutionsTableCard front={front} showScore={true} selectedId={null} selectEquation={() => {}} />);
    expect(screen.getByText("score")).toBeTruthy();
    expect(screen.getByText("1.500e+0")).toBeTruthy();
  });

  it("hides the score column for the linear loss scale", () => {
    render(<SolutionsTableCard front={front} showScore={false} selectedId={null} selectEquation={() => {}} />);
    expect(screen.queryByText("score")).toBeNull();
    expect(screen.queryByText("1.500e+0")).toBeNull();
  });
});
