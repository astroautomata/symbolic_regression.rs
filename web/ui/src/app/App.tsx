import React from "react";
import { useEffect } from "react";
import { useSessionStore } from "../state/sessionStore";
import { EnterData } from "./panes/EnterData";
import { ModelingTask } from "./panes/ModelingTask";
import { SearchSolutions } from "./panes/SearchSolutions";

export function App(): React.ReactElement {
  const tab = useSessionStore((s) => s.tab);
  const setTab = useSessionStore((s) => s.setTab);
  const loadWasmMetadata = useSessionStore((s) => s.loadWasmMetadata);

  useEffect(() => {
    void loadWasmMetadata();
  }, [loadWasmMetadata]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="title">PySR Online</div>
        <div className="tabs">
          <button className={tab === "data" ? "tab active" : "tab"} onClick={() => setTab("data")}>
            Data
          </button>
          <span className="tabSep" aria-hidden="true">
            →
          </span>
          <button className={tab === "configure" ? "tab active" : "tab"} onClick={() => setTab("configure")}>
            Configure
          </button>
          <span className="tabSep" aria-hidden="true">
            →
          </span>
          <button className={tab === "run" ? "tab active" : "tab"} onClick={() => setTab("run")}>
            Run
          </button>
        </div>
      </header>

      <main className="main">
        <div className="mainInner">
          {tab === "data" && <EnterData />}
          {tab === "configure" && <ModelingTask />}
          {tab === "run" && <SearchSolutions />}
        </div>
      </main>
    </div>
  );
}
