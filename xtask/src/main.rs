//! Repository tasks, run as `cargo xtask <task>`.

mod gen_parity;
mod mapping;
mod parity;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "usage: cargo xtask parity [--julia-repo <path>] [--require-done-through <N>] [--run]
       cargo xtask gen-parity --julia-repo <path>";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("parity") => parse_parity_args(&args[1..])
            .and_then(|(julia_repo, through, run)| parity::check(julia_repo.as_deref(), through, run)),
        Some("gen-parity") => match &args[1..] {
            [flag, repo] if flag == "--julia-repo" => gen_parity::regenerate(Path::new(repo)),
            _ => Err(USAGE.to_owned()),
        },
        _ => Err(USAGE.to_owned()),
    };
    result.unwrap_or_else(|err| {
        eprintln!("error: {err}");
        ExitCode::FAILURE
    })
}

/// `--julia-repo`, `--require-done-through` (default 0), and `--run`.
fn parse_parity_args(args: &[String]) -> Result<(Option<PathBuf>, i64, bool), String> {
    let (mut julia_repo, mut through, mut run) = (None, 0, false);
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--julia-repo" => julia_repo = Some(args.next().ok_or(USAGE)?.into()),
            "--require-done-through" => through = args.next().and_then(|n| n.parse().ok()).ok_or(USAGE)?,
            "--run" => run = true,
            _ => return Err(USAGE.to_owned()),
        }
    }
    Ok((julia_repo, through, run))
}
