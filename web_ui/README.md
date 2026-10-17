# sr_web

Minimal browser UI for `symbolic_regression_wasm` (runs symbolic regression in a WebWorker).

The Data pane accepts PySR unit strings for the target and a comma-separated list for the selected X columns. Blank fields leave units unset. Advanced hyperparameters include `dimensional_constraint_penalty` and `dimensionless_constants_only`; the solutions table shows unit annotations while copied equations remain parseable.

## Prereqs

- Rust target: `wasm32-unknown-unknown`
- `wasm-pack`
- Node.js + npm

## Dev

Build the wasm package into the web app:

```sh
cd symbolic_regression_wasm
wasm-pack build --target web --out-dir ../web_ui/src/pkg
```

Start the dev server:

```sh
cd web_ui
npm install
npm run dev
```
