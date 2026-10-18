use super::Span;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Keyword {
    Function,
    End,
    Return,
    Local,
    Where,
    True,
    False,
    For,
    While,
    If,
    Begin,
}

impl Keyword {
    fn from_ident(s: &str) -> Option<Self> {
        Some(match s {
            "function" => Self::Function,
            "end" => Self::End,
            "return" => Self::Return,
            "local" => Self::Local,
            "where" => Self::Where,
            "true" => Self::True,
            "false" => Self::False,
            "for" => Self::For,
            "while" => Self::While,
            "if" => Self::If,
            "begin" => Self::Begin,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Function => "function",
            Self::End => "end",
            Self::Return => "return",
            Self::Local => "local",
            Self::Where => "where",
            Self::True => "true",
            Self::False => "false",
            Self::For => "for",
            Self::While => "while",
            Self::If => "if",
            Self::Begin => "begin",
        }
    }

    /// Keywords that open a block closed by `end`.
    pub fn opens_block(self) -> bool {
        matches!(self, Self::Function | Self::For | Self::While | Self::If | Self::Begin)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Num(f64),
    Ident(String),
    Kw(Keyword),
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Semi,
    Newline,
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    Lt,
    Le,
    Gt,
    Ge,
    EqEq,
    Ne,
    AndAnd,
    OrOr,
    Bang,
    Question,
    Colon,
    ColonColon,
    SubType,
    Assign,
    Arrow,
    /// A construct the DSL rejects, with its error message. The parser reports it when it reaches
    /// this token, so errors come out in source order.
    Bad(String),
    Eof,
}

impl Tok {
    /// How the token reads in "expected X, found Y" messages.
    pub fn describe(&self) -> String {
        let s = match self {
            Self::Num(v) => return format!("`{v}`"),
            Self::Ident(s) => return format!("`{s}`"),
            Self::Kw(k) => k.as_str(),
            Self::LParen => "(",
            Self::RParen => ")",
            Self::LBracket => "[",
            Self::RBracket => "]",
            Self::LBrace => "{",
            Self::RBrace => "}",
            Self::Comma => ",",
            Self::Semi => ";",
            Self::Newline => return "end of line".to_string(),
            Self::Plus => "+",
            Self::Minus => "-",
            Self::Star => "*",
            Self::Slash => "/",
            Self::Caret => "^",
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
            Self::EqEq => "==",
            Self::Ne => "!=",
            Self::AndAnd => "&&",
            Self::OrOr => "||",
            Self::Bang => "!",
            Self::Question => "?",
            Self::Colon => ":",
            Self::ColonColon => "::",
            Self::SubType => "<:",
            Self::Assign => "=",
            Self::Arrow => "->",
            Self::Bad(_) => return "an unsupported construct".to_string(),
            Self::Eof => return "end of input".to_string(),
        };
        format!("`{s}`")
    }

    /// A newline right after this token continues the statement.
    fn continues_line(&self) -> bool {
        matches!(
            self,
            Self::Plus
                | Self::Minus
                | Self::Star
                | Self::Slash
                | Self::Caret
                | Self::Lt
                | Self::Le
                | Self::Gt
                | Self::Ge
                | Self::EqEq
                | Self::Ne
                | Self::AndAnd
                | Self::OrOr
                | Self::Bang
                | Self::Question
                | Self::Colon
                | Self::ColonColon
                | Self::SubType
                | Self::Comma
                | Self::Assign
                | Self::Arrow
                | Self::LParen
                | Self::LBracket
                | Self::LBrace
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    pub tok: Tok,
    pub span: Span,
}

/// Splits `src` into tokens, ending with [`Tok::Eof`]. Never fails: rejected constructs become
/// [`Tok::Bad`] tokens.
pub fn tokenize(src: &str) -> Vec<Token> {
    let mut lexer = Lexer {
        src,
        pos: 0,
        depth: 0,
        out: Vec::new(),
    };
    lexer.run();
    lexer.out
}

struct Lexer<'s> {
    src: &'s str,
    pos: usize,
    depth: usize,
    out: Vec<Token>,
}

fn is_ident_start(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}

impl Lexer<'_> {
    fn peek(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    fn peek_second(&self) -> Option<char> {
        self.src[self.pos..].chars().nth(1)
    }

    /// Digits, with Julia's `_` separators between them (`1_000`).
    fn eat_digits(&mut self) -> usize {
        let start = self.pos;
        while let Some(c) = self.peek() {
            let separator = c == '_' && self.pos > start && self.peek_second().is_some_and(|d| d.is_ascii_digit());
            if !(c.is_ascii_digit() || separator) {
                break;
            }
            self.pos += 1;
        }
        self.pos - start
    }

    fn push(&mut self, tok: Tok, start: usize) {
        match tok {
            Tok::LParen | Tok::LBracket | Tok::LBrace => self.depth += 1,
            Tok::RParen | Tok::RBracket | Tok::RBrace => self.depth = self.depth.saturating_sub(1),
            _ => {}
        }
        self.out.push(Token {
            tok,
            span: Span::new(start, self.pos),
        });
    }

    fn run(&mut self) {
        while let Some(c) = self.peek() {
            let start = self.pos;
            match c {
                '\n' => {
                    self.pos += 1;
                    self.newline(start);
                }
                '#' => {
                    while self.peek().is_some_and(|c| c != '\n') {
                        self.pos += self.peek().map_or(0, char::len_utf8);
                    }
                }
                c if c.is_whitespace() => self.pos += c.len_utf8(),
                '0'..='9' => self.number(),
                '.' if self.peek_second().is_some_and(|c| c.is_ascii_digit()) => self.number(),
                '.' => self.dot(),
                c if is_ident_start(c) => self.ident(),
                '"' => self.string(),
                _ => self.punct(c),
            }
        }
        let end = self.src.len();
        self.out.push(Token {
            tok: Tok::Eof,
            span: Span::new(end, end),
        });
    }

    fn newline(&mut self, start: usize) {
        if self.depth > 0 {
            return;
        }
        let Some(last) = self.out.last() else {
            return;
        };
        if matches!(last.tok, Tok::Newline | Tok::Semi) || last.tok.continues_line() {
            return;
        }
        self.out.push(Token {
            tok: Tok::Newline,
            span: Span::new(start, start + 1),
        });
    }

    fn number(&mut self) {
        let start = self.pos;
        self.eat_digits();
        if self.peek() == Some('.') {
            self.pos += 1;
            self.eat_digits();
        }
        // As in Julia, a number followed by `e`, `E`, or `f` is always one float literal, so `2e`
        // and `2exp(x)` are malformed numbers rather than `2 * e`.
        let mut float32 = false;
        let mut exponent_digits = true;
        if let Some(marker @ ('e' | 'E' | 'f')) = self.peek() {
            self.pos += 1;
            if matches!(self.peek(), Some('+' | '-')) {
                self.pos += 1;
            }
            float32 = marker == 'f';
            exponent_digits = self.eat_digits() > 0;
        }
        let text = &self.src[start..self.pos];
        let literal = text.replace('_', "").replace('f', "e");
        let value = if float32 {
            literal.parse::<f32>().map(f64::from).ok()
        } else {
            literal.parse::<f64>().ok()
        };
        let tok = match value {
            Some(v) if exponent_digits => Tok::Num(v),
            _ if !exponent_digits => Tok::Bad(format!(
                "invalid number `{text}`; an exponent needs digits, as in `1e-3`"
            )),
            _ => Tok::Bad(format!("invalid number `{text}`")),
        };
        self.push(tok, start);
    }

    fn ident(&mut self) {
        let start = self.pos;
        while let Some(c) = self.peek() {
            let continues = c.is_alphanumeric() || c == '_' || (c == '!' && self.peek_second() != Some('='));
            if !continues {
                break;
            }
            self.pos += c.len_utf8();
        }
        let text = &self.src[start..self.pos];
        let tok = match Keyword::from_ident(text) {
            Some(kw) => Tok::Kw(kw),
            None => Tok::Ident(text.to_string()),
        };
        self.push(tok, start);
    }

    /// A `.` that does not start a number: a function from another Julia package
    /// (`Primes.prime(`), broadcasting (`.+`, `f.(`), or property access (`a.b`).
    fn dot(&mut self) {
        let dot = self.pos;
        let qualifier = match self.out.last() {
            Some(Token {
                tok: Tok::Ident(name),
                span,
            }) if span.end == dot => Some(name.clone()),
            _ => None,
        };
        let start = match self.out.last() {
            Some(prev) if prev.span.end == dot && !matches!(prev.tok, Tok::Newline | Tok::Eof) => prev.span.start,
            _ => dot,
        };
        self.pos += 1;
        let tok = match self.peek() {
            Some(c) if is_ident_start(c) => {
                let member_start = self.pos;
                while self.peek().is_some_and(|c| c.is_alphanumeric() || c == '_') {
                    self.pos += self.peek().map_or(0, char::len_utf8);
                }
                let member = &self.src[member_start..self.pos];
                match qualifier {
                    Some(module) if self.peek() == Some('(') => Tok::Bad(format!(
                        "`{module}.{member}` is a function from an external Julia package, which is not supported"
                    )),
                    _ => Tok::Bad("property access (`a.b`) is not supported".to_string()),
                }
            }
            Some(c) if "(+-*/^<>=!&|%\\".contains(c) => Tok::Bad("broadcasting (`.`) is not supported".to_string()),
            _ => Tok::Bad("unexpected character `.`".to_string()),
        };
        let end = match &tok {
            Tok::Bad(msg) if msg.starts_with("broadcasting") => dot + 2,
            _ => self.pos,
        };
        self.out.push(Token {
            tok,
            span: Span::new(start, end),
        });
    }

    fn string(&mut self) {
        let start = self.pos;
        self.pos += 1;
        while let Some(c) = self.peek() {
            self.pos += c.len_utf8();
            match c {
                '\\' => self.pos += self.peek().map_or(0, char::len_utf8),
                '"' => break,
                _ => {}
            }
        }
        self.push(Tok::Bad("strings are not supported".to_string()), start);
    }

    fn punct(&mut self, c: char) {
        let start = self.pos;
        let next = self.peek_second();
        let (tok, len) = match (c, next) {
            ('(', _) => (Tok::LParen, 1),
            (')', _) => (Tok::RParen, 1),
            ('[', _) => (Tok::LBracket, 1),
            (']', _) => (Tok::RBracket, 1),
            ('{', _) => (Tok::LBrace, 1),
            ('}', _) => (Tok::RBrace, 1),
            (',', _) => (Tok::Comma, 1),
            (';', _) => (Tok::Semi, 1),
            ('+', _) => (Tok::Plus, 1),
            ('-', Some('>')) => (Tok::Arrow, 2),
            ('-', _) => (Tok::Minus, 1),
            ('*', _) => (Tok::Star, 1),
            ('/', _) => (Tok::Slash, 1),
            ('^', _) => (Tok::Caret, 1),
            ('<', Some('=')) => (Tok::Le, 2),
            ('<', Some(':')) => (Tok::SubType, 2),
            ('<', _) => (Tok::Lt, 1),
            ('>', Some('=')) => (Tok::Ge, 2),
            ('>', _) => (Tok::Gt, 1),
            ('=', Some('=')) => (Tok::EqEq, 2),
            ('=', _) => (Tok::Assign, 1),
            ('!', Some('=')) => (Tok::Ne, 2),
            ('!', _) => (Tok::Bang, 1),
            ('&', Some('&')) => (Tok::AndAnd, 2),
            ('|', Some('|')) => (Tok::OrOr, 2),
            ('?', _) => (Tok::Question, 1),
            (':', Some(':')) => (Tok::ColonColon, 2),
            (':', _) => (Tok::Colon, 1),
            _ => (Tok::Bad(format!("unexpected character `{c}`")), c.len_utf8()),
        };
        self.pos += len;
        self.push(tok, start);
    }
}
