use ndarray::array;

use crate::prelude::*;
use crate::progress_bars::{RenderOptions, format_hall_of_fame};
use crate::{Guess, equation_search};

#[test]
fn display_names_and_target_names_are_distinct_from_csv_names() {
    let dataset = Dataset::with_weights_and_names(array![[1., 2.]], array![1., 2.], None, vec!["x".into()])
        .with_display_names(vec!["α".into()], Some("energy".into()))
        .unwrap();
    let options = Options::<f64, BuiltinOpsF64, 2> {
        operators: BuiltinOpsF64::new().from_names(["+"]).unwrap(),
        guesses: vec![vec![Guess::from("x")]],
        niterations: 0,
        populations: 1,
        population_size: 5,
        save_to_file: false,
        progress: false,
        verbosity: Some(0),
        input_stream: None,
        ..Default::default()
    };
    let result = equation_search(std::slice::from_ref(&dataset), &options).unwrap();
    let table = format_hall_of_fame(
        &result.halls_of_fame[0],
        &options,
        100,
        usize::MAX,
        RenderOptions {
            ansi: false,
            loss_scale: options.loss_scale,
        },
        5,
        None,
        Some(&dataset.display_variable_names),
        &dataset.y_variable_name,
    )
    .unwrap();
    assert!(table.contains("energy = α"));
    let serialized = result.halls_of_fame[0].members().next().unwrap().expr.string(
        &options,
        dynamic_expressions::strings::StringTreeOptions {
            variable_names: Some(&dataset.variable_names),
            ..Default::default()
        },
    );
    assert_eq!(serialized, "x");
    let batch = Dataset::make_batch_buffer(&dataset, 1);
    assert_eq!(batch.display_variable_names, ["α"]);
    assert_eq!(batch.y_variable_name, "energy");
}
#[test]
fn dataset_target_defaults_and_multi_output_names() {
    let data = Dataset::with_weights_and_names(array![[1., 2.]], array![1., 2.], None, vec!["y".into()]);
    assert_eq!(data.display_variable_names, ["y"]);
    assert_eq!(data.y_variable_name, "target");
    let outputs = Dataset::multi_output(array![[1., 2.]], array![[1., 2.], [3., 4.]], None, vec!["x".into()]);
    assert_eq!(outputs[0].y_variable_name, "y₁");
    assert_eq!(outputs[1].y_variable_name, "y₂");
    assert!(data.with_display_names(vec![], None).is_err());
}
#[test]
fn print_precision_is_used_for_terminal_equations_only() {
    let data = Dataset::with_weights_and_names(array![[1., 2.]], array![1., 2.], None, vec!["x".into()]);
    let options = Options::<f64, BuiltinOpsF64, 2> {
        operators: BuiltinOpsF64::new().from_names(["+"]).unwrap(),
        guesses: vec![vec![Guess::from("x + 1.23456789")]],
        niterations: 0,
        populations: 1,
        population_size: 5,
        save_to_file: false,
        progress: false,
        verbosity: Some(0),
        input_stream: None,
        should_optimize_constants: false,
        ..Default::default()
    };
    let result = equation_search(std::slice::from_ref(&data), &options).unwrap();
    let table = format_hall_of_fame(
        &result.halls_of_fame[0],
        &options,
        100,
        usize::MAX,
        RenderOptions {
            ansi: false,
            loss_scale: options.loss_scale,
        },
        3,
        None,
        Some(&data.display_variable_names),
        &data.y_variable_name,
    )
    .unwrap();
    assert!(table.contains("x + 1.23"));
    assert!(!table.contains("1.23456789"));
    assert!(
        result.halls_of_fame[0]
            .members()
            .next()
            .unwrap()
            .expr
            .string(&options, Default::default())
            .contains("1.23456789")
    );
}
