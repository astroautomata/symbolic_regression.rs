//! tools/parity_tests.toml, which maps every SymbolicRegression.jl testitem to its Rust port.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use toml_edit::{DocumentMut, Item};

pub const JULIA_REPO_URL: &str = "https://github.com/astroautomata/SymbolicRegression.jl";
pub const JULIA_COMMIT: &str = "f97764fb898a0279ca434a615d4e40f52fdb9469";
pub const EXPECTED_TESTITEMS: usize = 287;
pub const EXPECTED_FILES: usize = 84;

const HEADER: &str = r#"# One entry per @testitem in SymbolicRegression.jl (test/unit and test/integration) at julia_commit.
#
#   status   "todo" (not ported yet) | "done" (ported) | "excluded" (will not be ported)
#   pr       todo and done only: the feature PR that owns the testitem
#   rust     done only: Rust tests, as "<crate>/<path>.rs::<test fn>"
#   excluded excluded only: the reason
#   note     optional: what the port does not carry over, and why
#
# `cargo xtask gen-parity` adds and refreshes entries; edit status/pr/rust/excluded/note by hand.
# `cargo xtask parity` fails on structural errors: a testitem missing from this file, an entry that
# is not a testitem, inconsistent fields, or a Rust test that does not exist (and, with --run, one
# that fails). It prints done/todo/excluded counts per PR. "todo" entries fail only under
# --require-done-through N, which rejects any todo whose pr <= N; CI raises N as each PR lands.
"#;

/// A testitem's (file, name).
pub type Key = (String, String);

#[derive(Clone, Debug)]
pub struct Entry {
    pub file: String,
    pub line: i64,
    pub name: String,
    pub pr: Option<i64>,
    pub status: String,
    pub rust: Vec<String>,
    pub excluded: Option<String>,
    pub note: Option<String>,
}

impl Entry {
    pub fn key(&self) -> Key {
        (self.file.clone(), self.name.clone())
    }
}

pub struct Mapping {
    pub julia_commit: Option<String>,
    pub entries: Vec<Entry>,
}

pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask is in the repository root")
        .to_path_buf()
}

pub fn toml_path() -> PathBuf {
    root().join("tools/parity_tests.toml")
}

pub fn load(path: &Path) -> Result<Mapping, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn parse(text: &str) -> Result<Mapping, String> {
    let doc: DocumentMut = text.parse().map_err(|e: toml_edit::TomlError| e.to_string())?;
    let julia_commit = doc.get("julia_commit").and_then(Item::as_str).map(str::to_owned);
    let tables = match doc.get("testitem") {
        None => {
            return Ok(Mapping {
                julia_commit,
                entries: vec![],
            });
        }
        Some(item) => item
            .as_array_of_tables()
            .ok_or("`testitem` must be an array of tables")?,
    };
    let entries = tables
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let string = |key: &str| t.get(key).and_then(Item::as_str).map(str::to_owned);
            let required = |key: &str| string(key).ok_or(format!("testitem {i}: `{key}` must be a string"));
            let rust = match t.get("rust") {
                None => vec![],
                Some(item) => item
                    .as_array()
                    .and_then(|refs| refs.iter().map(|r| r.as_str().map(str::to_owned)).collect())
                    .ok_or(format!("testitem {i}: `rust` must be an array of strings"))?,
            };
            Ok(Entry {
                file: required("file")?,
                line: t
                    .get("line")
                    .and_then(Item::as_integer)
                    .ok_or(format!("testitem {i}: `line` must be an integer"))?,
                name: required("name")?,
                pr: t.get("pr").and_then(Item::as_integer),
                status: required("status")?,
                rust,
                excluded: string("excluded"),
                note: string("note"),
            })
        })
        .collect::<Result<_, String>>()?;
    Ok(Mapping { julia_commit, entries })
}

pub fn write(entries: &[Entry], path: &Path) -> Result<(), String> {
    let mut out = vec![
        HEADER.to_owned(),
        format!("julia_repo = {}", toml_str(JULIA_REPO_URL)),
        format!("julia_commit = {}", toml_str(JULIA_COMMIT)),
        String::new(),
    ];
    for e in entries {
        out.push("[[testitem]]".to_owned());
        out.push(format!("file = {}", toml_str(&e.file)));
        out.push(format!("line = {}", e.line));
        out.push(format!("name = {}", toml_str(&e.name)));
        if let Some(pr) = e.pr {
            out.push(format!("pr = {pr}"));
        }
        out.push(format!("status = {}", toml_str(&e.status)));
        if !e.rust.is_empty() {
            let refs: Vec<String> = e.rust.iter().map(|r| toml_str(r)).collect();
            out.push(format!("rust = [{}]", refs.join(", ")));
        }
        for (key, value) in [("excluded", &e.excluded), ("note", &e.note)] {
            if let Some(value) = value {
                out.push(format!("{key} = {}", toml_str(value)));
            }
        }
        out.push(String::new());
    }
    std::fs::write(path, out.join("\n")).map_err(|e| format!("{}: {e}", path.display()))
}

/// A TOML basic string, escaped as JSON escapes it.
fn toml_str(s: &str) -> String {
    let mut out = String::from('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if c < ' ' => write!(out, "\\u{:04x}", c as u32).unwrap(),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Every (file, line, name) of an `@testitem` under test/unit and test/integration at `JULIA_COMMIT`.
pub fn extract(repo: &Path) -> Result<Vec<(String, i64, String)>, String> {
    let files = git(
        repo,
        &[
            "ls-tree",
            "-r",
            "--name-only",
            JULIA_COMMIT,
            "--",
            "test/unit",
            "test/integration",
        ],
    )?;
    let mut items = vec![];
    for path in files.split_whitespace().filter(|f| f.ends_with(".jl")) {
        let source = git(repo, &["show", &format!("{JULIA_COMMIT}:{path}")])?;
        for (i, line) in source.lines().enumerate() {
            let Some(name) = testitem_name(line) else { continue };
            if name.contains(['\\', '$']) {
                return Err(format!(
                    "{path}:{}: testitem name uses escapes or interpolation: {name:?}",
                    i + 1
                ));
            }
            items.push((path.to_owned(), i as i64 + 1, name.to_owned()));
        }
    }
    let n_files = items.iter().map(|(path, ..)| path).collect::<BTreeSet<_>>().len();
    if items.len() != EXPECTED_TESTITEMS || n_files != EXPECTED_FILES {
        return Err(format!(
            "expected {EXPECTED_TESTITEMS} testitems in {EXPECTED_FILES} files, found {} in {n_files}",
            items.len()
        ));
    }
    if items
        .iter()
        .map(|(path, _, name)| (path, name))
        .collect::<BTreeSet<_>>()
        .len()
        != items.len()
    {
        return Err("duplicate (file, name) testitem keys".to_owned());
    }
    Ok(items)
}

/// The name in `@testitem "name" ...`, the macro call at the start of `line`.
fn testitem_name(line: &str) -> Option<&str> {
    let rest = line.trim_start().strip_prefix("@testitem")?;
    let quoted = rest.trim_start();
    if quoted.len() == rest.len() {
        return None;
    }
    let body = quoted.strip_prefix('"')?;
    let mut escaped = false;
    for (i, c) in body.char_indices() {
        match c {
            _ if escaped => escaped = false,
            '\\' => escaped = true,
            '"' => return Some(&body[..i]),
            _ => {}
        }
    }
    None
}

fn git(repo: &Path, args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .map_err(|e| format!("git: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    String::from_utf8(out.stdout).map_err(|e| format!("git {}: {e}", args.join(" ")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn testitem_names_are_read_from_the_macro_call() {
        assert_eq!(
            testitem_name(r#"@testitem "Test whether \"x\" works" tags = [:part1] begin"#),
            Some(r#"Test whether \"x\" works"#)
        );
        assert_eq!(testitem_name(r#"    @testitem   "indented""#), Some("indented"));
        assert_eq!(testitem_name(r#"@testitem"no space""#), None);
        assert_eq!(testitem_name(r#"# @testitem "comment""#), None);
        assert_eq!(testitem_name(r#"@testitem "unterminated"#), None);
    }

    #[test]
    fn written_mapping_reads_back_unchanged() {
        let path = std::env::temp_dir().join(format!("parity_tests_{}.toml", std::process::id()));
        let original = std::fs::read_to_string(toml_path()).unwrap();
        write(&load(&toml_path()).unwrap().entries, &path).unwrap();
        let rewritten = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(original == rewritten, "writing the checked-in mapping changes it");
    }
}
