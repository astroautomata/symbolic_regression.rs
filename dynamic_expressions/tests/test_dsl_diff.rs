use dynamic_expressions::dsl::diff::differentiate;
use dynamic_expressions::dsl::ir::{Graph, Node, NodeId};
use dynamic_expressions::dsl::lower::{Lowered, lower};
use dynamic_expressions::dsl::parse_definition;

fn ir_string(graph: &Graph, id: NodeId) -> String {
    match graph.node(id) {
        Node::Const(bits) => format!("{}", f64::from_bits(*bits)),
        Node::Arg(i) => format!("#{i}"),
        Node::Op { kernel, args } => {
            let args: Vec<String> = args.iter().map(|a| ir_string(graph, *a)).collect();
            format!("({} {})", kernel.name(), args.join(" "))
        }
        Node::Partial { kernel, idx, args } => {
            let args: Vec<String> = args.iter().map(|a| ir_string(graph, *a)).collect();
            format!("(d{idx}{} {})", kernel.name(), args.join(" "))
        }
    }
}

fn lower_with(src: &str, customs: &[Lowered]) -> Lowered {
    let def = parse_definition(src).unwrap_or_else(|e| panic!("{src:?}: {e}"));
    lower(&def, customs).unwrap_or_else(|e| panic!("{src:?}: {e}"))
}

fn deriv(src: &str, wrt: u16) -> String {
    let l = lower_with(src, &[]);
    let mut graph = l.graph.clone();
    let d = differentiate(&mut graph, l.root, wrt);
    ir_string(&graph, d)
}

#[test]
fn diff_arithmetic_is_symbolic() {
    assert_eq!(deriv("f(x, y) = x + y", 0), "1");
    assert_eq!(deriv("f(x, y) = x - y", 1), "-1");
    assert_eq!(deriv("f(x, y) = x * y", 0), "#1");
    assert_eq!(deriv("f(x, y) = x * y", 1), "#0");
    assert_eq!(deriv("f(x) = x * x", 0), "(add #0 #0)");
    assert_eq!(deriv("f(x) = 1 / x", 0), "(neg (div 1 (mul #0 #0)))");
    assert_eq!(deriv("f(x, y) = x / y", 0), "(div 1 #1)");
    assert_eq!(deriv("f(x) = -x", 0), "-1");
    assert_eq!(deriv("f(x, y) = 3 * x + y", 0), "3");
    assert_eq!(deriv("f(x, y) = y", 0), "0");
    assert_eq!(deriv("f() = 2", 0), "0");
}

#[test]
fn diff_other_kernels_use_partials() {
    assert_eq!(deriv("f(x) = sin(x)", 0), "(d0sin #0)");
    assert_eq!(deriv("f(x) = x^2", 0), "(d0pow #0 2)");
    assert_eq!(deriv("f(x, y) = x^y", 1), "(d1pow #0 #1)");
    assert_eq!(deriv("f(x) = exp(sin(x))", 0), "(mul (d0exp (sin #0)) (d0sin #0))");
    assert_eq!(
        deriv("f(x, y) = fma(x, y, x)", 0),
        "(add (d0fma #0 #1 #0) (d2fma #0 #1 #0))"
    );
    assert_eq!(deriv("f(x, y) = min(x, 2 * y)", 1), "(mul (d1min #0 (mul 2 #1)) 2)");
}

#[test]
fn diff_ifelse_differentiates_each_branch() {
    assert_eq!(deriv("f(x) = x > 0 ? x : -x", 0), "(ifelse (greater #0 0) 1 -1)");
    assert_eq!(
        deriv("f(x, y) = ifelse(x > y, x * y, 0)", 0),
        "(ifelse (greater #0 #1) #1 0)"
    );
    assert_eq!(deriv("f(x, y) = x > 0 ? y : 1", 0), "0");
}

#[test]
fn diff_comparisons_and_logic_are_zero() {
    for src in [
        "f(x) = x > 0",
        "f(x) = x <= 1",
        "f(x) = x == 2",
        "f(x) = x != 2",
        "f(x) = x > 0 && x < 1",
        "f(x) = !(x > 0) || x >= 3",
        "f(x) = greater(x, 0)",
    ] {
        assert_eq!(deriv(src, 0), "0", "{src}");
    }
    assert_eq!(deriv("f(x) = x * (x > 0)", 0), "(greater #0 0)");
}
