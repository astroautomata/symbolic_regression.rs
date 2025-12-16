import React from "react";
import type { EquationSummary } from "../../../types/srTypes";
import { formatSci } from "./plotUtils";

export function SolutionsTableCard(props: {
  front: EquationSummary[];
  selectedId: string | null;
  selectEquation: (sel: { id: string; complexity: number }) => void;
}): React.ReactElement {
  return (
    <div className="card gridCell">
      <div className="cardTitle">Current solutions</div>
      <div className="tableWrap">
        <table className="table" data-testid="solutions-table">
          <thead>
            <tr>
              <th>size</th>
              <th>loss</th>
              <th>equation</th>
            </tr>
          </thead>
          <tbody>
            {props.front
              .slice()
              .reverse()
              .map((m) => (
                <tr
                  key={m.id}
                  className={m.id === props.selectedId ? "selectedRow" : ""}
                  onClick={() => props.selectEquation({ id: m.id, complexity: m.complexity })}
                  data-testid={`solution-row-${m.id}`}
                >
                  <td className="mono">{m.complexity}</td>
                  <td className="mono">{formatSci(m.loss)}</td>
                  <td className="mono">{m.equation}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
