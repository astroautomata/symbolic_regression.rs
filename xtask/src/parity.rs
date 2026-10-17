//! `cargo xtask parity`: check tools/parity_tests.toml against SymbolicRegression.jl and the Rust tests.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::{Command, ExitCode};

use toml_edit::DocumentMut;

use crate::mapping::{self, EXPECTED_FILES, EXPECTED_TESTITEMS, Entry, JULIA_COMMIT, Key, Mapping};

const STATUSES: [&str; 3] = ["todo", "done", "excluded"];

pub fn check(julia_repo: Option<&Path>, require_done_through: i64, run: bool) -> Result<ExitCode, String> {
    let mapping = mapping::load(&mapping::toml_path())?;
    let julia = match julia_repo {
        Some(repo) => Some(
            mapping::extract(repo)?
                .into_iter()
                .map(|(file, _, name)| (file, name))
                .collect(),
        ),
        None => None,
    };
    let mut errors = structural_errors(&mapping, julia.as_ref());
    if errors.is_empty() {
        println!("{}", summary(&mapping.entries));
        errors.extend(
            mapping
                .entries
                .iter()
                .filter(|e| e.status == "todo" && e.pr.is_some_and(|pr| pr <= require_done_through))
                .map(|e| format!("todo in PR {}: {} | {}", e.pr.expect("filtered on pr"), e.file, e.name)),
        );
    }
    if errors.is_empty() && run {
        errors.extend(run_errors(&mapping.entries)?);
    }
    for err in &errors {
        eprintln!("error: {err}");
    }
    Ok(if errors.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

/// Errors in the mapping itself, and with `julia`, testitems it lacks or has in excess.
fn structural_errors(mapping: &Mapping, julia: Option<&BTreeSet<Key>>) -> Vec<String> {
    let mut errors = vec![];
    if mapping.julia_commit.as_deref() != Some(JULIA_COMMIT) {
        let found = mapping
            .julia_commit
            .as_ref()
            .map_or("None".to_owned(), |c| format!("'{c}'"));
        errors.push(format!("julia_commit is {found}, expected '{JULIA_COMMIT}'"));
    }
    let keys: Vec<Key> = mapping.entries.iter().map(Entry::key).collect();
    for (i, key) in keys.iter().enumerate() {
        if !keys[..i].contains(key) && keys[i + 1..].contains(key) {
            errors.push(format!("duplicate entry: {} | {}", key.0, key.1));
        }
    }
    let n_files = keys.iter().map(|(file, _)| file).collect::<BTreeSet<_>>().len();
    if keys.len() != EXPECTED_TESTITEMS || n_files != EXPECTED_FILES {
        errors.push(format!(
            "expected {EXPECTED_TESTITEMS} entries in {EXPECTED_FILES} files, found {} in {n_files}",
            keys.len()
        ));
    }
    if let Some(julia) = julia {
        let mapped: BTreeSet<Key> = keys.into_iter().collect();
        errors.extend(
            julia
                .difference(&mapped)
                .map(|(f, n)| format!("missing from tools/parity_tests.toml: {f} | {n}")),
        );
        errors.extend(
            mapped
                .difference(julia)
                .map(|(f, n)| format!("not a testitem at {}: {f} | {n}", &JULIA_COMMIT[..8])),
        );
    }
    for e in &mapping.entries {
        let place = format!("{} | {}", e.file, e.name);
        if !STATUSES.contains(&e.status.as_str()) {
            errors.push(format!(
                "{place}: status '{}' is not one of ('todo', 'done', 'excluded')",
                e.status
            ));
            continue;
        }
        if e.status != "excluded" && !e.pr.is_some_and(|pr| (1..=8).contains(&pr)) {
            errors.push(format!("{place}: {} entries need pr = 1..8", e.status));
        }
        if (e.status == "excluded") != e.excluded.as_ref().is_some_and(|reason| !reason.is_empty()) {
            errors.push(format!(
                "{place}: `excluded` reason must be set exactly when status is excluded"
            ));
        }
        if (e.status == "done") == e.rust.is_empty() {
            errors.push(format!(
                "{place}: `rust` tests must be listed exactly when status is done"
            ));
        }
        errors.extend(
            e.rust
                .iter()
                .filter_map(|r| rust_ref_error(r))
                .map(|err| format!("{place}: {err}")),
        );
    }
    errors
}

/// A `rust` entry: `<crate>/tests/<file>.rs::<fn>` or `<crate>/src/<path>.rs::<fn>`.
struct RustRef<'a> {
    krate: &'a str,
    /// The file within the crate, `tests/...` or `src/...`.
    path: &'a str,
    func: &'a str,
}

impl<'a> RustRef<'a> {
    fn parse(r: &'a str) -> Option<Self> {
        let is_word = |c: char| c.is_alphanumeric() || c == '_';
        let (file, func) = r.split_once("::")?;
        let (krate, path) = file.split_once('/')?;
        let stem = path
            .strip_prefix("tests/")
            .or_else(|| path.strip_prefix("src/"))?
            .strip_suffix(".rs")?;
        let valid = !krate.is_empty()
            && krate.chars().all(|c| c.is_ascii_lowercase() || c == '_')
            && !stem.is_empty()
            && stem.chars().all(|c| is_word(c) || c == '/')
            && !func.is_empty()
            && func.chars().all(is_word);
        valid.then_some(RustRef { krate, path, func })
    }

    /// The `cargo test` arguments that select this test's target, and the test's name within it.
    fn cargo_invocation(&self) -> Result<(Vec<String>, String), String> {
        let manifest_path = mapping::root().join(self.krate).join("Cargo.toml");
        let manifest =
            std::fs::read_to_string(&manifest_path).map_err(|e| format!("{}: {e}", manifest_path.display()))?;
        let manifest: DocumentMut = manifest
            .parse()
            .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
        let package = manifest
            .get("package")
            .and_then(|p| p.get("name"))
            .and_then(|n| n.as_str())
            .ok_or(format!("{}: no package name", manifest_path.display()))?
            .to_owned();
        let stem = self.path.strip_suffix(".rs").expect("parsed paths end in .rs");
        Ok(match stem.strip_prefix("tests/") {
            Some(test) => {
                let target = test.rsplit('/').next().expect("rsplit yields at least one piece");
                (
                    vec!["-p".into(), package, "--test".into(), target.into()],
                    self.func.to_owned(),
                )
            }
            None => {
                let module = stem
                    .strip_prefix("src/")
                    .expect("parsed paths are under tests/ or src/");
                let module = module.strip_suffix("/mod").unwrap_or(module).replace('/', "::");
                (
                    vec!["-p".into(), package, "--lib".into()],
                    format!("{module}::{}", self.func),
                )
            }
        })
    }
}

fn rust_ref_error(r: &str) -> Option<String> {
    let Some(rust_ref) = RustRef::parse(r) else {
        return Some(format!(
            "bad Rust test reference '{r}'; use <crate>/tests/<file>.rs::<fn> or <crate>/src/<path>.rs::<fn>"
        ));
    };
    let path = mapping::root().join(rust_ref.krate).join(rust_ref.path);
    if !path.is_file() {
        return Some(format!(
            "Rust test file does not exist: {}/{}",
            rust_ref.krate, rust_ref.path
        ));
    }
    match std::fs::read_to_string(&path) {
        Err(e) => Some(format!("{}: {e}", path.display())),
        Ok(source) if !has_test_fn(&source, rust_ref.func) => Some(format!(
            "no #[test] fn {} in {}/{}",
            rust_ref.func, rust_ref.krate, rust_ref.path
        )),
        Ok(_) => None,
    }
}

/// Whether `source` has `#[test]`, then any other attributes, then `fn <name>(`, optionally `pub`.
fn has_test_fn(source: &str, name: &str) -> bool {
    source.match_indices("#[test]").any(|(i, attr)| {
        let mut rest = source[i + attr.len()..].trim_start();
        while let Some((_, after)) = rest.strip_prefix("#[").and_then(|r| r.split_once(']')) {
            rest = after.trim_start();
        }
        let rest = match rest.strip_prefix("pub") {
            Some(after) if after.starts_with(char::is_whitespace) => after.trim_start(),
            _ => rest,
        };
        rest.strip_prefix("fn")
            .filter(|r| r.starts_with(char::is_whitespace))
            .and_then(|r| r.trim_start().strip_prefix(name))
            .is_some_and(|r| r.trim_start().starts_with('('))
    })
}

/// Runs every mapped Rust test, one `cargo test` per test target.
fn run_errors(entries: &[Entry]) -> Result<Vec<String>, String> {
    let mut targets: BTreeMap<Vec<String>, BTreeSet<String>> = BTreeMap::new();
    for r in entries.iter().flat_map(|e| &e.rust) {
        let (args, name) = RustRef::parse(r)
            .expect("structural_errors checked every reference")
            .cargo_invocation()?;
        targets.entry(args).or_default().insert(name);
    }
    let mut errors = vec![];
    for (args, names) in &targets {
        let mut cmd_args: Vec<&str> = vec!["test"];
        cmd_args.extend(args.iter().map(String::as_str));
        cmd_args.extend(["--all-features", "--", "--exact"]);
        cmd_args.extend(names.iter().map(String::as_str));
        let cmd = format!("cargo {}", cmd_args.join(" "));
        println!("$ {cmd}");
        let out = Command::new("cargo")
            .args(&cmd_args)
            .current_dir(mapping::root())
            .output()
            .map_err(|e| format!("{cmd}: {e}"))?;
        let stdout = String::from_utf8_lossy(&out.stdout);
        let passed: BTreeSet<&str> = stdout
            .lines()
            .filter_map(|line| line.strip_prefix("test ")?.strip_suffix(" ... ok"))
            .filter(|name| !name.contains(char::is_whitespace))
            .collect();
        let failed: Vec<&String> = names.iter().filter(|n| !passed.contains(n.as_str())).collect();
        errors.extend(
            failed
                .iter()
                .map(|n| format!("Rust test did not pass: {} {n}", args.join(" "))),
        );
        if !out.status.success() && failed.is_empty() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let tail = stderr.char_indices().rev().nth(1999).map_or(0, |(i, _)| i);
            errors.push(format!("cargo exited with {}: {cmd}\n{}", out.status, &stderr[tail..]));
        }
    }
    Ok(errors)
}

/// done/todo/excluded counts per PR, then in total.
fn summary(entries: &[Entry]) -> String {
    let mut by_pr: BTreeMap<String, [usize; 3]> = BTreeMap::new();
    let mut total = [0; 3];
    for e in entries {
        let status = STATUSES
            .iter()
            .position(|s| *s == e.status)
            .expect("structural_errors checked every status");
        let pr = match e.pr {
            Some(pr) if e.status != "excluded" => pr.to_string(),
            _ => "-".to_owned(),
        };
        by_pr.entry(pr).or_default()[status] += 1;
        total[status] += 1;
    }
    let mut lines = vec!["pr  done  todo  excluded".to_owned()];
    for (pr, [todo, done, excluded]) in by_pr.iter().map(|(pr, c)| (pr.as_str(), *c)).chain([("all", total)]) {
        lines.push(format!("{pr:<3} {done:>4}  {todo:>4}  {excluded:>8}"));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load() -> Mapping {
        mapping::load(&mapping::toml_path()).unwrap()
    }

    /// The testitems the checked-in mapping lists, standing in for SymbolicRegression.jl's.
    fn julia_testitems() -> BTreeSet<Key> {
        load().entries.iter().map(Entry::key).collect()
    }

    fn first_todo(mapping: &mut Mapping) -> &mut Entry {
        mapping.entries.iter_mut().find(|e| e.status == "todo").unwrap()
    }

    #[test]
    fn checked_in_mapping_is_clean() {
        assert_eq!(
            structural_errors(&load(), Some(&julia_testitems())),
            Vec::<String>::new()
        );
    }

    #[test]
    fn missing_testitem_is_an_error() {
        let mut mapping = load();
        let dropped = mapping.entries.remove(0);
        let errors = structural_errors(&mapping, Some(&julia_testitems()));
        let expected = format!(
            "missing from tools/parity_tests.toml: {} | {}",
            dropped.file, dropped.name
        );
        assert!(errors.contains(&expected), "{errors:?}");
    }

    #[test]
    fn renamed_testitem_is_an_error() {
        let mut mapping = load();
        mapping.entries[0].name += " (renamed)";
        let errors = structural_errors(&mapping, Some(&julia_testitems()));
        assert!(
            errors.iter().any(|e| e.starts_with("not a testitem at f97764fb")),
            "{errors:?}"
        );
    }

    #[test]
    fn missing_rust_test_file_is_an_error() {
        let mut mapping = load();
        let entry = first_todo(&mut mapping);
        entry.status = "done".into();
        entry.rust = vec!["dynamic_expressions/tests/no_such_file.rs::no_such_test".into()];
        let errors = structural_errors(&mapping, None);
        assert!(
            errors.iter().any(|e| e.contains("Rust test file does not exist")),
            "{errors:?}"
        );
    }

    #[test]
    fn missing_rust_test_fn_is_an_error() {
        let mut mapping = load();
        let entry = first_todo(&mut mapping);
        entry.status = "done".into();
        entry.rust = vec!["dynamic_expressions/tests/test_eval.rs::no_such_test".into()];
        let errors = structural_errors(&mapping, None);
        assert!(
            errors.iter().any(|e| e.contains("no #[test] fn no_such_test")),
            "{errors:?}"
        );
    }

    #[test]
    fn done_without_rust_is_an_error() {
        let mut mapping = load();
        first_todo(&mut mapping).status = "done".into();
        let errors = structural_errors(&mapping, None);
        assert!(
            errors.iter().any(|e| e.contains("`rust` tests must be listed")),
            "{errors:?}"
        );
    }
}
