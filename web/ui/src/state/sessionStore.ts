import { create } from "zustand";
import * as Papa from "papaparse";
import initWasm, { builtin_operator_registry, default_search_options } from "../pkg/symbolic_regression_wasm.js";
import type {
  EquationPoint,
  EquationSummary,
  SearchSnapshot,
  WasmEvalResult,
  WasmOpInfo,
  WasmSearchOptions,
  WasmSplitIndices
} from "../types/srTypes";

export type TabKey = "data" | "clean" | "task" | "search";

type ParsedDataset = {
  headers: string[];
  rows: number[][];
};

type SearchRuntime = {
  status: "idle" | "initializing" | "ready" | "running" | "paused" | "done" | "error";
  error: string | null;
  split: WasmSplitIndices | null;
  snapshot: SearchSnapshot | null;
  front: EquationSummary[];
  selectedId: string | null;
  selectedComplexity: number | null;
  evalByKey: Record<string, WasmEvalResult>;
};

type SessionState = {
  tab: TabKey;

  wasmLoaded: boolean;
  operatorRegistry: WasmOpInfo[];
  defaultOptions: WasmSearchOptions | null;

  csvText: string;
  parsed: ParsedDataset | null;

  options: WasmSearchOptions | null;
  unaryOps: string[];
  binaryOps: string[];
  ternaryOps: string[];

  runtime: SearchRuntime;

  setTab: (tab: TabKey) => void;
  loadWasmMetadata: () => Promise<void>;
  setCsvText: (csvText: string) => void;
  parseCsv: () => void;
  setOptionsPatch: (patch: Partial<WasmSearchOptions>) => void;

  toggleOp: (arity: 1 | 2 | 3, name: string) => void;
  applyPreset: (preset: "basic" | "trig" | "explog" | "all") => void;

  setRuntime: (patch: Partial<SearchRuntime>) => void;
  setSnapshot: (snap: SearchSnapshot) => void;
  setFront: (front: EquationSummary[]) => void;
  setSelection: (id: string | null, complexity: number | null) => void;
  setEvalResult: (memberId: string, which: "train" | "val", result: WasmEvalResult) => void;
};

const DEFAULT_CSV = `x0,x1,x2,x3,x4,y
2.066531,-0.476571,-2.754929,-1.207267,2.903399,-0.754701
2.689256,-2.472590,-2.532614,1.245805,2.879404,-0.745495
0.214802,2.065321,-2.892979,1.004816,-1.603409,2.201418
-2.774622,0.330249,2.633970,-1.034287,2.219611,-0.014409
2.404120,1.582256,-0.167322,-2.395793,-0.394968,-0.962298
-1.850552,-0.168717,-2.398315,-1.719650,-1.128322,-2.079073
-0.020073,1.357735,2.815159,1.861303,-0.137941,-0.125028
-0.355398,1.467173,-0.255768,-0.405912,1.109319,-0.661907
2.479085,2.799638,0.951533,2.412421,-1.138914,6.232683
1.378990,-0.838550,-2.672266,-1.676604,-1.560383,-1.128941
0.644120,2.777435,2.244292,1.327387,-2.099100,5.888947
0.916488,-0.037551,2.191859,2.790597,1.859721,-0.262792
-0.763173,1.386895,-1.499269,1.457695,-1.740702,0.550708
0.249286,0.620131,-2.733398,-1.253074,-2.565287,-0.589299
1.863411,-1.892869,2.252245,0.524128,-2.112411,2.846828
-1.485012,1.532805,-1.858703,-2.460088,-1.459193,0.687636
1.232865,2.131407,2.891211,-0.823875,-2.899198,2.542589
-1.451388,0.895216,0.269321,2.801934,-1.101426,-0.148713
2.144169,0.617068,0.111746,2.518368,0.428337,-1.536687
-1.170207,1.261114,-1.680271,2.530700,-2.871634,0.511758
0.937061,2.849679,0.621449,0.677118,1.226925,5.532206
2.434374,1.062702,1.503803,-2.158352,0.925962,-0.415381
-2.615961,-2.419039,-0.384905,2.381290,2.503698,6.048525
0.602684,0.969518,2.596097,-2.681471,-0.143618,-1.860180
1.239323,-1.636620,-2.497553,2.598675,0.410902,1.161655
0.765324,-0.665365,0.913350,-2.404820,0.694156,-0.854871
-1.244346,1.435056,-1.393001,1.620965,2.819434,1.122449
0.656316,1.985768,-1.042447,2.683397,-1.697902,1.862603
0.182865,0.297299,-0.500532,-1.516102,-2.803250,-0.218787
-1.531699,-0.153271,0.676103,-0.284815,1.479933,-0.884059
1.110441,1.908302,-0.180125,2.239199,2.659298,1.733959
-1.808747,1.124197,-2.493524,1.764394,-2.069082,0.279727
-0.438388,2.484967,-2.971004,2.295548,1.468971,4.216744
1.891029,1.676807,-0.185622,-0.482753,0.969184,0.781983
-2.746168,0.063508,-1.774107,1.455342,1.150856,-1.282105
2.637592,2.619122,0.579484,-0.519003,2.446145,4.885714
-0.310631,1.349002,1.065380,1.034678,-0.590310,0.220056
1.186606,2.036643,-0.002279,-2.150628,2.633734,2.196882
0.884568,-2.245210,0.104284,1.952008,-2.499186,3.519778
0.206816,0.688060,2.527820,-1.502919,2.875222,0.005874
-1.641576,-1.898098,0.228519,-1.945310,0.448169,0.140090
1.616825,-2.168044,2.209961,0.023499,0.191885,2.638885
1.877262,-2.382779,0.266178,-2.664767,2.218026,0.508090
2.023775,-1.335313,0.039539,2.257584,2.320380,1.572484
-0.463105,-1.613898,0.315546,0.950272,1.328429,-1.520147
2.880014,-0.354593,0.180572,-2.458187,-2.845417,-2.761488
-1.784950,1.786902,2.304159,0.811681,1.692882,-0.768243
2.058153,0.400932,-0.641407,1.280795,0.252785,-0.972708
0.720391,2.270477,-2.327416,0.246686,2.308647,4.171304
-0.916526,-1.927051,2.339099,-0.730941,2.185518,-0.393477
-1.910938,0.448774,2.174350,-0.719692,1.650994,-1.310603
1.903094,-0.098581,1.158797,-1.090163,0.290756,-1.006686
2.050985,-1.880890,-0.799903,-0.354123,2.504632,1.838497
0.558801,0.607306,1.790427,0.341695,0.910087,-0.596159
2.889877,-0.658152,-1.969992,0.365266,0.311570,-1.910859
2.434753,-1.696777,0.212833,1.039033,-0.528482,0.184490
-1.463325,0.278211,-2.733339,0.809352,2.350839,-0.376636
2.338080,1.489041,2.973710,-0.350616,0.362331,-0.115859
2.311746,-1.814482,0.151260,2.021385,-1.406980,3.050931
-2.842180,2.364633,-2.637782,2.841594,-0.961115,4.787348
1.411177,2.594475,-1.490985,-0.281153,2.040731,5.571319
0.887937,-1.209483,-2.093787,2.936046,-1.372324,0.302078
1.871466,-1.974100,-0.668457,-2.333135,0.143158,2.267526
2.732855,-0.435976,1.191007,1.089154,1.058735,-0.203541
-0.566832,-2.177888,0.165186,-2.402270,-1.724022,-2.440902
-0.354286,0.892645,0.448337,2.487053,2.323652,0.346005
1.026612,1.399548,0.137350,1.777082,-2.439210,1.969806
1.621843,1.463546,-2.726468,2.666668,0.857839,-1.205665
-1.815270,0.379967,2.617001,-0.781411,-1.098296,-1.678780
0.213574,0.494131,-1.072069,-2.318810,-2.785597,-1.832490
2.953446,-1.503980,-1.622375,-1.675109,1.560597,-0.263343
-2.187775,0.369178,0.582434,2.454221,2.504582,-0.543287
0.255898,-2.289488,1.334030,-0.330236,-1.661987,4.274684
2.265699,-0.542884,2.556886,1.134575,2.710411,-0.481325
2.053440,-2.971093,-2.428912,-1.932655,1.296221,-0.273209
-1.776816,0.722680,1.899093,1.621557,-0.149614,-0.857520
-2.766696,-1.405657,2.622158,-2.770889,0.087101,0.435919
-0.479155,1.670244,2.831418,-0.439546,-1.279551,0.790489
-2.217831,1.534741,2.764931,-2.946809,0.222021,0.478790
2.095721,2.431672,-2.264942,0.426442,0.078507,4.292257
-1.964654,2.577202,0.997666,-0.280506,0.089707,4.790429
0.773005,-2.745212,1.157147,0.019620,0.664333,-0.856374
2.064604,1.946032,0.916401,1.612076,-2.768213,2.205383
0.120334,2.830727,-0.433020,1.030312,0.468560,6.166550
1.727507,1.725702,1.908715,1.038170,0.515801,1.110320
-0.267476,-2.771619,0.000676,1.595802,-1.679629,-0.776651
-0.921227,-0.561688,2.155366,0.036420,-2.602688,0.539777
-2.903193,0.468518,-2.579783,0.630320,2.854539,-0.753336
-2.553537,-0.876012,-1.759502,-0.189575,2.855480,-0.057803
0.429603,-1.598622,-0.104002,2.891535,0.500770,0.121552
-0.900531,0.020701,-1.799782,0.811838,0.301076,-0.913509
2.567670,2.075926,1.190209,-2.008228,0.842619,2.265408
-1.068630,2.240552,2.309152,0.496885,-1.435542,4.830149
2.504095,-2.217701,0.800646,-2.130918,1.805324,2.134499
1.392421,1.481283,-0.131642,2.845395,1.208142,0.190445
2.483293,1.928299,-1.723980,-1.660217,0.270420,1.814649
-0.271850,-1.861167,0.064932,-2.915381,0.688461,-0.541129
-2.230887,1.819485,1.315874,2.393886,-0.166065,1.683864
0.236130,2.755739,2.472132,-0.918823,1.035221,5.785936
-2.395543,0.769290,-0.300418,-0.132482,2.623061,0.080416
2.766116,-2.613718,-1.973610,1.541008,-0.865556,4.838567
2.345733,2.937243,-2.782895,-0.178198,0.667346,6.923771
-0.392629,2.342114,1.455792,-1.138062,-2.812799,3.936236
-2.331806,1.951586,-0.102528,0.543364,-0.050333,1.840379
0.491263,1.127011,2.969159,-1.487978,0.254305,0.897576
0.432395,-2.409066,0.583320,0.038132,-2.510415,3.385980
0.148385,2.170659,-1.365963,2.482410,0.444850,3.424950
2.353593,1.968950,1.971468,-2.550873,-0.931697,2.913058
0.624433,-0.189178,0.886681,-1.465989,0.568123,-1.547054
0.862263,1.628141,0.458616,1.150099,-1.392258,1.627303
0.538175,-0.893528,2.945540,2.574906,-2.062191,-0.316677
-1.989185,-2.545867,2.278704,0.865324,-2.391333,4.918729
`;

function keyForEval(memberId: string, which: "train" | "val"): string {
  return `${memberId}:${which}`;
}

function toggleInList(list: string[], value: string): string[] {
  return list.includes(value) ? list.filter((v) => v !== value) : [...list, value];
}

export const useSessionStore = create<SessionState>((set, get) => ({
  tab: "data",

  wasmLoaded: false,
  operatorRegistry: [],
  defaultOptions: null,

  csvText: DEFAULT_CSV,
  parsed: null,

  options: null,
  unaryOps: [],
  binaryOps: [],
  ternaryOps: [],

  runtime: {
    status: "idle",
    error: null,
    split: null,
    snapshot: null,
    front: [],
    selectedId: null,
    selectedComplexity: null,
    evalByKey: {}
  },

  setTab: (tab) => set({ tab }),

  loadWasmMetadata: async () => {
    if (get().wasmLoaded) return;
    await initWasm();
    const [ops, defaults] = await Promise.all([builtin_operator_registry(), default_search_options()]);

    const operatorRegistry = ops as WasmOpInfo[];
    const defaultOptions = defaults as WasmSearchOptions;
    set({
      wasmLoaded: true,
      operatorRegistry,
      defaultOptions,
      options: defaultOptions
    });

    // Default operator selection (a safe "basic" set).
    get().applyPreset("basic");
  },

  setCsvText: (csvText) => set({ csvText }),

  parseCsv: () => {
    const { csvText, options } = get();
    if (!options) return;

    const res = Papa.parse(csvText, {
      skipEmptyLines: true,
      dynamicTyping: true
    });
    if (res.errors.length > 0) {
      set((s) => ({
        runtime: { ...s.runtime, status: "error", error: res.errors[0]?.message ?? "CSV parse error" }
      }));
      return;
    }

    const raw = res.data as unknown[];
    if (raw.length === 0) return;

    let headers: string[] = [];
    let dataRows: unknown[] = raw;
    if (options.has_headers) {
      const h0 = raw[0] as unknown;
      if (!Array.isArray(h0)) {
        set((s) => ({
          runtime: { ...s.runtime, status: "error", error: "expected header row to be an array" }
        }));
        return;
      }
      headers = h0.map((v) => String(v));
      dataRows = raw.slice(1);
    }

    const rows: number[][] = [];
    for (const r of dataRows) {
      if (!Array.isArray(r)) continue;
      const row = r.map((v) => Number(v));
      if (row.some((v) => !Number.isFinite(v))) continue;
      rows.push(row);
    }
    if (rows.length === 0) {
      set((s) => ({
        runtime: { ...s.runtime, status: "error", error: "no numeric data rows parsed" }
      }));
      return;
    }

    const nCols = rows[0].length;
    if (headers.length !== nCols) {
      headers = Array.from({ length: nCols }, (_, i) => `col_${i}`);
    }

    // Defaults: y is last col, X is all others.
    const y = options.y_column ?? (nCols - 1);
    const weights = options.weights_column ?? null;
    const x =
      options.x_columns ??
      Array.from({ length: nCols }, (_, i) => i).filter((i) => i !== y && i !== weights);

    set({
      parsed: { headers, rows },
      options: {
        ...options,
        y_column: y,
        weights_column: weights,
        x_columns: x
      }
    });
  },

  setOptionsPatch: (patch) =>
    set((s) => ({
      options: s.options ? { ...s.options, ...patch } : s.options
    })),

  toggleOp: (arity, name) =>
    set((s) => {
      if (arity === 1) return { unaryOps: toggleInList(s.unaryOps, name) };
      if (arity === 2) return { binaryOps: toggleInList(s.binaryOps, name) };
      return { ternaryOps: toggleInList(s.ternaryOps, name) };
    }),

  applyPreset: (preset) =>
    set(() => {
      // Default presets intentionally avoid unary negation ("neg") and all ternary ops.
      // Initial load uses "basic": only +, -, *, /.
      const basicUnary: string[] = [];
      const basicBinary = ["add", "sub", "mul", "div"];
      const trigUnary = ["sin", "cos"];
      const explogUnary = ["exp", "log"];

      if (preset === "basic") return { unaryOps: basicUnary, binaryOps: basicBinary, ternaryOps: [] };
      if (preset === "trig") return { unaryOps: [...basicUnary, ...trigUnary], binaryOps: basicBinary, ternaryOps: [] };
      if (preset === "explog")
        return { unaryOps: [...basicUnary, ...explogUnary], binaryOps: basicBinary, ternaryOps: [] };
      return {}; // "all" handled in UI (requires registry)
    }),

  setRuntime: (patch) =>
    set((s) => ({
      runtime: { ...s.runtime, ...patch }
    })),

  setSnapshot: (snap) =>
    set((s) => ({
      runtime: { ...s.runtime, snapshot: snap }
    })),

  setFront: (front) =>
    set((s) => ({
      runtime: { ...s.runtime, front }
    })),

  setSelection: (id, complexity) =>
    set((s) => ({
      runtime: { ...s.runtime, selectedId: id, selectedComplexity: complexity }
    })),

  setEvalResult: (memberId, which, result) =>
    set((s) => ({
      runtime: {
        ...s.runtime,
        evalByKey: { ...s.runtime.evalByKey, [keyForEval(memberId, which)]: result }
      }
    }))
}));

export function getSelectedSummary(front: EquationSummary[], best: EquationSummary | null, selectedId: string | null) {
  if (selectedId == null) return best;
  return front.find((m) => m.id === selectedId) ?? (best?.id === selectedId ? best : null);
}

export function getParetoPoints(snap: SearchSnapshot | null): EquationPoint[] {
  return snap?.pareto_points ?? [];
}
