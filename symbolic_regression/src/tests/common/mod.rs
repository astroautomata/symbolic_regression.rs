use dynamic_expressions::operator_enum::builtin::*;
use dynamic_expressions::opset;

pub type T = f64;
pub const D: usize = 2;

opset! {
    pub TestOps for f64 { Sin, Cos, Exp, Log, Neg, Add, Sub, Mul, Div }
}

/// Builds an expression from Julia-style infix such as `1.0 * cos(x1 - 0.5 * x2)`: `xN` is feature
/// `N` (1-based), numbers are constants, and calls and `+ - * / ^` are looked up in `ops`.
pub fn infix<Ops: dynamic_expressions::OperatorSet, const N: usize>(
    ops: &Ops,
    src: &str,
) -> dynamic_expressions::PostfixExpr<T, Ops, N> {
    let mut p = Infix {
        ops,
        src: src.as_bytes(),
        pos: 0,
        nodes: Vec::new(),
        consts: Vec::new(),
    };
    p.sum();
    assert_eq!(p.pos, p.src.len(), "trailing input in `{src}`");
    dynamic_expressions::PostfixExpr::new(p.nodes, p.consts, dynamic_expressions::Metadata::default())
}

struct Infix<'a, Ops> {
    ops: &'a Ops,
    src: &'a [u8],
    pos: usize,
    nodes: Vec<dynamic_expressions::PNode>,
    consts: Vec<T>,
}

impl<Ops: dynamic_expressions::OperatorSet> Infix<'_, Ops> {
    fn skip_space(&mut self) {
        while self.src.get(self.pos) == Some(&b' ') {
            self.pos += 1;
        }
    }

    fn eat(&mut self, c: u8) -> bool {
        self.skip_space();
        let found = self.src.get(self.pos) == Some(&c);
        self.pos += usize::from(found);
        found
    }

    fn op(&mut self, name: &str, arity: u8) {
        let op = self
            .ops
            .lookup_with_arity(name, arity)
            .unwrap_or_else(|e| panic!("`{name}`: {e:?}"));
        self.nodes.push(dynamic_expressions::PNode::Op { arity, op: op.id });
    }

    fn sum(&mut self) {
        self.product();
        loop {
            if self.eat(b'+') {
                self.product();
                self.op("+", 2);
            } else if self.eat(b'-') {
                self.product();
                self.op("-", 2);
            } else {
                return;
            }
        }
    }

    fn product(&mut self) {
        self.power();
        loop {
            if self.eat(b'*') {
                self.power();
                self.op("*", 2);
            } else if self.eat(b'/') {
                self.power();
                self.op("/", 2);
            } else {
                return;
            }
        }
    }

    fn power(&mut self) {
        self.atom();
        if self.eat(b'^') {
            self.power();
            self.op("^", 2);
        }
    }

    fn atom(&mut self) {
        self.skip_space();
        if self.eat(b'(') {
            self.sum();
            assert!(self.eat(b')'), "missing `)`");
            return;
        }
        let start = self.pos;
        while self
            .src
            .get(self.pos)
            .is_some_and(|c| c.is_ascii_alphanumeric() || b"._".contains(c))
        {
            self.pos += 1;
        }
        let token = std::str::from_utf8(&self.src[start..self.pos]).unwrap();
        if let Ok(v) = token.parse::<T>() {
            self.nodes.push(dynamic_expressions::PNode::Const {
                idx: self.consts.len() as u16,
            });
            self.consts.push(v);
        } else if let Some(i) = token.strip_prefix('x').and_then(|i| i.parse::<u16>().ok()) {
            self.nodes.push(dynamic_expressions::PNode::Var { feature: i - 1 });
        } else {
            assert!(self.eat(b'('), "unknown token `{token}`");
            let mut arity = 0;
            loop {
                self.sum();
                arity += 1;
                if !self.eat(b',') {
                    break;
                }
            }
            assert!(self.eat(b')'), "missing `)` after `{token}`");
            self.op(token, arity);
        }
    }
}
