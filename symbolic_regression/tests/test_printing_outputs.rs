use ndarray::array;
use symbolic_regression::prelude::*;
use symbolic_regression::{
    Guess, equation_search, format_hall, format_member, format_search_state, split_string, wrap_equation_string,
};

fn fixture() -> (
    Options<f64, BuiltinOpsF64, 2>,
    symbolic_regression::PopMember<f64, BuiltinOpsF64, 2>,
) {
    let options = Options {
        operators: BuiltinOpsF64::new().from_names(&["+", "^"]).unwrap(),
        niterations: 0,
        populations: 1,
        population_size: 5,
        tournament_selection_n: 2,
        maxsize: 7,
        deterministic: true,
        should_optimize_constants: false,
        progress: false,
        save_to_file: false,
        input_stream: None,
        guesses: vec![vec![Guess::from("x^2.0 + 1.5")]],
        ..Default::default()
    };
    let data = Dataset::with_weights_and_names(array![[1., 2., 3.]], array![2., 3., 4.], None, vec!["x".into()]);
    let result = equation_search(&[data], &options).unwrap();
    let mut member = result.halls_of_fame[0].members().next().unwrap().clone();
    member.cost = 1.;
    (options, member)
}
#[test]
fn pretty_print_member() {
    let (options, member) = fixture();
    assert_eq!(
        format_member(&member, &options, &["x".into()]),
        "PopMember(tree = ((x ^ 2.0) + 1.5), loss = 16.25, cost = 1.0)"
    );
    let mut unchanged = options.clone();
    unchanged.operators = BuiltinOpsF64::new().from_names(&["sub", "/"]).unwrap();
    assert_eq!(
        format_member(&member, &options, &["x".into()]),
        format_member(&member, &unchanged, &["x".into()])
    );
}
#[test]
fn pretty_print_hall_of_fame() {
    let (options, member) = fixture();
    let mut hall = symbolic_regression::HallOfFame::new(7);
    hall.best_by_complexity[5] = Some(member);
    assert_eq!(
        format_hall(&hall, &options, &["x".into()]),
        "HallOfFame{...}:\n    .exists[1] = false\n    .members[1] = undef\n    .exists[2] = false\n    .members[2] = undef\n    .exists[3] = false\n    .members[3] = undef\n    .exists[4] = false\n    .members[4] = undef\n    .exists[5] = true\n    .members[5] = PopMember(tree = ((x ^ 2.0) + 1.5), loss = 16.25, cost = 1.0)\n    .exists[6] = false\n    .members[6] = undef\n    .exists[7] = false\n    .members[7] = undef"
    );
}
#[test]
fn printing_utilities() {
    assert_eq!(split_string("abc\ndefg", 3), ["abc", "\nde", "fg"]);
    assert_eq!(split_string("αβγδ", 3), ["αβγ", "δ"]);
    assert_eq!(
        wrap_equation_string("cos(x) + 1.5387438743 - y^2", 0, 15),
        "cos(x) + 1....\n5387438743 ...\n- y^2\n"
    );
    let multiline = "(\nB = ( -0.012549, 0.0086419, 0.6175 )\nF_d = (-0.051546) * v\n)";
    assert_eq!(
        wrap_equation_string(multiline, 4, 1000),
        "(\n    B = ( -0.012549, 0.0086419, 0.6175 )\n    F_d = (-0.051546) * v\n    )\n"
    );
    assert!(wrap_equation_string(multiline, 0, 10).starts_with("(\n"));
    assert_eq!(
        wrap_equation_string(multiline, 0, 12),
        "(\nB = ( -0...\n.012549,...\n 0.00864...\n19, 0.61...\n75 )\nF_d = (-...\n0.051546...\n) * v\n)\n"
    );
}
#[test]
fn print_search_state_does_not_round_partial_iterations_up() {
    let partial = format_search_state(31, 30, 31);
    assert!(partial.contains("Progress: 0 / 1 total iterations"));
    assert!(partial.contains("(3.226%)"));
    let complete = format_search_state(31, 0, 31);
    assert!(complete.contains("Progress: 1 / 1 total iterations"));
    assert!(complete.contains("(100.000%)"));
}
