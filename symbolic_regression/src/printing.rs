use std::fmt::{Display, Write};

use dynamic_expressions::OperatorSet;
use dynamic_expressions::strings::{StringTreeOptions, julia_float};
use num_traits::Float;

use crate::{HallOfFame, Options, PopMember};

/// Splits at Unicode character boundaries, retaining explicit newlines.
pub fn split_string(s: &str, width: usize) -> Vec<String> {
    assert!(width > 0, "split width must be positive");
    let chars: Vec<_> = s.chars().collect();
    if chars.is_empty() {
        return vec![String::new()];
    }
    chars.chunks(width).map(|chunk| chunk.iter().collect()).collect()
}
/// Wraps equations with continuation dots and indents all but the first line.
pub fn wrap_equation_string(equation: &str, left_cols_width: usize, terminal_width: usize) -> String {
    let width = terminal_width
        .checked_sub(left_cols_width + 4)
        .filter(|&n| n > 0)
        .expect("terminal width must leave space for the equation");
    let mut result = String::new();
    let mut pad = false;
    for piece in equation.split('\n') {
        let pieces = split_string(piece, width);
        for (i, piece) in pieces.iter().enumerate() {
            if pad {
                result.push_str(&" ".repeat(left_cols_width));
            }
            result.push_str(piece);
            if i + 1 < pieces.len() {
                result.push_str("...");
            }
            result.push('\n');
            pad = true;
        }
    }
    result
}
/// Formats a member at full precision, as SymbolicRegression.jl's plain-text display does.
pub fn format_member<T: Float + Display, Ops: OperatorSet, const D: usize>(
    member: &PopMember<T, Ops, D>,
    options: &Options<T, Ops, D>,
    variable_names: &[String],
) -> String {
    let equation = member.expr.string(
        options,
        StringTreeOptions {
            variable_names: Some(variable_names),
            julia_constants: true,
            pretty: true,
            ..Default::default()
        },
    );
    format!(
        "PopMember(tree = ({equation}), loss = {}, cost = {})",
        julia_float(member.loss.to_string().parse().expect("float loss")),
        julia_float(member.cost.to_string().parse().expect("float cost"))
    )
}
/// Formats every complexity slot, including absent members.
pub fn format_hall<T: Float + Display, Ops: OperatorSet, const D: usize>(
    hall: &HallOfFame<T, Ops, D>,
    options: &Options<T, Ops, D>,
    variable_names: &[String],
) -> String {
    let mut result = String::from("HallOfFame{...}:");
    for (complexity, member) in hall.best_by_complexity.iter().enumerate().skip(1) {
        write!(
            result,
            "\n    .exists[{complexity}] = {}\n    .members[{complexity}] = {}",
            member.is_some(),
            member
                .as_ref()
                .map_or_else(|| "undef".into(), |m| format_member(m, options, variable_names))
        )
        .unwrap();
    }
    result
}
/// Counts completed iterations only after all their population cycles have completed.
pub fn format_search_state(total_cycles: usize, cycles_remaining: usize, populations: usize) -> String {
    assert!(populations > 0, "a search needs populations");
    let elapsed = total_cycles.saturating_sub(cycles_remaining);
    let percent = if total_cycles == 0 {
        100.0
    } else {
        100.0 * elapsed as f64 / total_cycles as f64
    };
    format!(
        "Progress: {} / {} total iterations ({percent:.3}%)",
        elapsed / populations,
        total_cycles / populations
    )
}

pub(crate) fn subscriptify(index: usize) -> String {
    index
        .to_string()
        .chars()
        .map(|c| match c {
            '0' => '₀',
            '1' => '₁',
            '2' => '₂',
            '3' => '₃',
            '4' => '₄',
            '5' => '₅',
            '6' => '₆',
            '7' => '₇',
            '8' => '₈',
            '9' => '₉',
            _ => unreachable!("decimal digit"),
        })
        .collect()
}
