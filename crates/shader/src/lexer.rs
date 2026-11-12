//! Tokenizer: WGSL tokens, nested comments, numeric literals and template-list discovery.

use crate::Error;

/// A 1-based source position (line, column in characters).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Span {
    pub line: u32,
    pub col: u32,
}

impl Span {
    pub fn error(self, message: impl Into<String>) -> Error {
        Error {
            line: self.line,
            column: self.col,
            message: message.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IntLit {
    Abstract(i64),
    I32(i32),
    U32(u32),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FloatLit {
    Abstract(f64),
    F32(f32),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Ident(String),
    Int(IntLit),
    Float(FloatLit),
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Semi,
    Colon,
    Comma,
    Dot,
    At,
    Arrow,
    Underscore,
    Assign,
    EqEq,
    NotEq,
    Lt,
    Le,
    Gt,
    Ge,
    Shl,
    Shr,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Amp,
    AmpAmp,
    Pipe,
    PipePipe,
    Caret,
    Tilde,
    Bang,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    PercentEq,
    AmpEq,
    PipeEq,
    CaretEq,
    ShlEq,
    ShrEq,
    PlusPlus,
    MinusMinus,
    /// A `<` that WGSL's template-list discovery identified as opening a template list.
    TemplateStart,
    /// A `>` that closes a template list (split off `>>`, `>=` or `>>=` when needed).
    TemplateEnd,
    Eof,
}

impl Tok {
    pub fn describe(&self) -> String {
        match self {
            Tok::Ident(s) => format!("'{s}'"),
            Tok::Int(_) | Tok::Float(_) => "a number".into(),
            Tok::Eof => "end of input".into(),
            other => format!("'{}'", other.text()),
        }
    }

    pub fn text(&self) -> &'static str {
        match self {
            Tok::Ident(_) => "identifier",
            Tok::Int(_) | Tok::Float(_) => "number",
            Tok::LParen => "(",
            Tok::RParen => ")",
            Tok::LBrace => "{",
            Tok::RBrace => "}",
            Tok::LBracket => "[",
            Tok::RBracket => "]",
            Tok::Semi => ";",
            Tok::Colon => ":",
            Tok::Comma => ",",
            Tok::Dot => ".",
            Tok::At => "@",
            Tok::Arrow => "->",
            Tok::Underscore => "_",
            Tok::Assign => "=",
            Tok::EqEq => "==",
            Tok::NotEq => "!=",
            Tok::Lt | Tok::TemplateStart => "<",
            Tok::Le => "<=",
            Tok::Gt | Tok::TemplateEnd => ">",
            Tok::Ge => ">=",
            Tok::Shl => "<<",
            Tok::Shr => ">>",
            Tok::Plus => "+",
            Tok::Minus => "-",
            Tok::Star => "*",
            Tok::Slash => "/",
            Tok::Percent => "%",
            Tok::Amp => "&",
            Tok::AmpAmp => "&&",
            Tok::Pipe => "|",
            Tok::PipePipe => "||",
            Tok::Caret => "^",
            Tok::Tilde => "~",
            Tok::Bang => "!",
            Tok::PlusEq => "+=",
            Tok::MinusEq => "-=",
            Tok::StarEq => "*=",
            Tok::SlashEq => "/=",
            Tok::PercentEq => "%=",
            Tok::AmpEq => "&=",
            Tok::PipeEq => "|=",
            Tok::CaretEq => "^=",
            Tok::ShlEq => "<<=",
            Tok::ShrEq => ">>=",
            Tok::PlusPlus => "++",
            Tok::MinusMinus => "--",
            Tok::Eof => "end of input",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Token {
    pub tok: Tok,
    pub span: Span,
}

/// Tokenize `src`, then mark template-list delimiters exactly as WGSL's template-list
/// discovery algorithm does, so `array<vec2<f32>, N>` and `a < b && c > d` both parse.
pub fn tokenize(src: &str) -> Result<Vec<Token>, Error> {
    let mut lx = Lexer {
        chars: src.chars().collect(),
        pos: 0,
        line: 1,
        col: 1,
    };
    let mut out = Vec::new();
    loop {
        lx.skip_trivia()?;
        let span = lx.span();
        let Some(c) = lx.peek(0) else {
            out.push(Token {
                tok: Tok::Eof,
                span,
            });
            break;
        };
        let tok =
            if c.is_ascii_digit() || (c == '.' && lx.peek(1).is_some_and(|d| d.is_ascii_digit())) {
                lx.number()?
            } else if c == '_' || c.is_alphabetic() {
                lx.ident()?
            } else {
                lx.punct()?
            };
        out.push(Token { tok, span });
    }
    discover_templates(&mut out);
    Ok(out)
}

struct Lexer {
    chars: Vec<char>,
    pos: usize,
    line: u32,
    col: u32,
}

impl Lexer {
    fn peek(&self, n: usize) -> Option<char> {
        self.chars.get(self.pos + n).copied()
    }

    fn span(&self) -> Span {
        Span {
            line: self.line,
            col: self.col,
        }
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek(0)?;
        self.pos += 1;
        if c == '\n' || (c == '\r' && self.peek(0) != Some('\n')) {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(c)
    }

    fn skip_trivia(&mut self) -> Result<(), Error> {
        loop {
            match self.peek(0) {
                Some(c) if is_blank(c) => {
                    self.bump();
                }
                Some('/') if self.peek(1) == Some('/') => {
                    while let Some(c) = self.peek(0) {
                        if c == '\n' || c == '\r' {
                            break;
                        }
                        self.bump();
                    }
                }
                Some('/') if self.peek(1) == Some('*') => {
                    let start = self.span();
                    self.bump();
                    self.bump();
                    let mut depth = 1usize;
                    while depth > 0 {
                        match (self.peek(0), self.peek(1)) {
                            (None, _) => return Err(start.error("unterminated block comment")),
                            (Some('/'), Some('*')) => {
                                self.bump();
                                self.bump();
                                depth += 1;
                            }
                            (Some('*'), Some('/')) => {
                                self.bump();
                                self.bump();
                                depth -= 1;
                            }
                            _ => {
                                self.bump();
                            }
                        }
                    }
                }
                _ => return Ok(()),
            }
        }
    }

    fn ident(&mut self) -> Result<Tok, Error> {
        let span = self.span();
        let mut s = String::new();
        while let Some(c) = self.peek(0) {
            if c == '_' || c.is_alphanumeric() {
                s.push(c);
                self.bump();
            } else {
                break;
            }
        }
        if s == "_" {
            return Ok(Tok::Underscore);
        }
        if s.starts_with("__") {
            return Err(span.error(format!(
                "identifier '{s}' is invalid: identifiers must not start with two underscores"
            )));
        }
        Ok(Tok::Ident(s))
    }

    fn number(&mut self) -> Result<Tok, Error> {
        let span = self.span();
        let tok = if self.peek(0) == Some('0') && matches!(self.peek(1), Some('x' | 'X')) {
            self.hex_number(span)?
        } else {
            self.decimal_number(span)?
        };
        if let Some(c) = self.peek(0)
            && (c == '_' || c.is_alphanumeric() || c == '.')
        {
            return Err(span.error(format!("invalid numeric literal: unexpected '{c}'")));
        }
        Ok(tok)
    }

    fn digits(&mut self, out: &mut String, hex: bool) {
        while let Some(c) = self.peek(0) {
            if c.is_ascii_digit() || (hex && c.is_ascii_hexdigit()) {
                out.push(c);
                self.bump();
            } else {
                break;
            }
        }
    }

    fn decimal_number(&mut self, span: Span) -> Result<Tok, Error> {
        let mut text = String::new();
        self.digits(&mut text, false);
        let int_part_len = text.len();
        let mut is_float = false;
        if self.peek(0) == Some('.') {
            // `2.` and `2.5` are floats; a '.' followed by an identifier is not part of a number,
            // but WGSL's longest-match rule makes `2.x` a malformed literal anyway.
            is_float = true;
            text.push('.');
            self.bump();
            self.digits(&mut text, false);
        }
        if matches!(self.peek(0), Some('e' | 'E')) {
            let has_digits = match self.peek(1) {
                Some('+' | '-') => self.peek(2).is_some_and(|c| c.is_ascii_digit()),
                Some(c) => c.is_ascii_digit(),
                None => false,
            };
            if has_digits {
                is_float = true;
                text.push('e');
                self.bump();
                if let Some(sign @ ('+' | '-')) = self.peek(0) {
                    text.push(sign);
                    self.bump();
                }
                self.digits(&mut text, false);
            } else {
                return Err(span.error("invalid numeric literal: exponent has no digits"));
            }
        }
        let suffix = match self.peek(0) {
            Some(c @ ('f' | 'h' | 'i' | 'u')) => {
                self.bump();
                Some(c)
            }
            _ => None,
        };
        if !is_float && int_part_len > 1 && text.starts_with('0') {
            return Err(span.error("invalid numeric literal: leading zeros are not allowed"));
        }
        if suffix == Some('h') {
            return Err(span.error("f16 literals are not supported (Ganbate has no f16)"));
        }
        if is_float || suffix == Some('f') {
            if is_float && matches!(suffix, Some('i' | 'u')) {
                return Err(span.error("invalid numeric literal: integer suffix on a float"));
            }
            if suffix == Some('f') {
                let v: f32 = text
                    .parse()
                    .map_err(|_| span.error("invalid floating-point literal"))?;
                if !v.is_finite() {
                    return Err(span.error("floating-point literal is out of range for f32"));
                }
                return Ok(Tok::Float(FloatLit::F32(v)));
            }
            let v: f64 = text
                .parse()
                .map_err(|_| span.error("invalid floating-point literal"))?;
            if !v.is_finite() {
                return Err(span.error("floating-point literal is out of range"));
            }
            return Ok(Tok::Float(FloatLit::Abstract(v)));
        }
        let value: u64 = text
            .parse()
            .map_err(|_| span.error("integer literal is out of range"))?;
        int_literal(value, suffix, span)
    }

    fn hex_number(&mut self, span: Span) -> Result<Tok, Error> {
        self.bump();
        self.bump();
        let mut int_digits = String::new();
        self.digits(&mut int_digits, true);
        let mut frac_digits = String::new();
        let mut is_float = false;
        if self.peek(0) == Some('.') {
            is_float = true;
            self.bump();
            self.digits(&mut frac_digits, true);
        }
        let mut exponent: i64 = 0;
        if matches!(self.peek(0), Some('p' | 'P')) {
            is_float = true;
            self.bump();
            let mut neg = false;
            if let Some(sign @ ('+' | '-')) = self.peek(0) {
                neg = sign == '-';
                self.bump();
            }
            let mut e = String::new();
            self.digits(&mut e, false);
            if e.is_empty() {
                return Err(span.error("invalid hexadecimal float: exponent has no digits"));
            }
            exponent = e
                .parse::<i64>()
                .map_err(|_| span.error("hexadecimal float exponent out of range"))?;
            if neg {
                exponent = -exponent;
            }
        }
        if int_digits.is_empty() && frac_digits.is_empty() {
            return Err(span.error("invalid hexadecimal literal: no digits"));
        }
        if is_float {
            let suffix = match self.peek(0) {
                Some(c @ ('f' | 'h')) => {
                    self.bump();
                    Some(c)
                }
                _ => None,
            };
            if suffix == Some('h') {
                return Err(span.error("f16 literals are not supported (Ganbate has no f16)"));
            }
            let all = format!("{int_digits}{frac_digits}");
            let trimmed = all.trim_start_matches('0');
            if trimmed.len() > 15 {
                return Err(span.error("hexadecimal float has too many significant digits"));
            }
            let mantissa = if trimmed.is_empty() {
                0
            } else {
                u64::from_str_radix(trimmed, 16)
                    .map_err(|_| span.error("invalid hexadecimal float"))?
            };
            let exp = exponent - 4 * frac_digits.len() as i64;
            let exp = exp.clamp(-2000, 2000) as i32;
            let v = (mantissa as f64) * 2f64.powi(exp);
            if !v.is_finite() {
                return Err(span.error("floating-point literal is out of range"));
            }
            return Ok(Tok::Float(if suffix == Some('f') {
                if !(v as f32).is_finite() {
                    return Err(span.error("floating-point literal is out of range for f32"));
                }
                FloatLit::F32(v as f32)
            } else {
                FloatLit::Abstract(v)
            }));
        }
        let suffix = match self.peek(0) {
            Some(c @ ('i' | 'u')) => {
                self.bump();
                Some(c)
            }
            _ => None,
        };
        let value = u64::from_str_radix(&int_digits, 16)
            .map_err(|_| span.error("integer literal is out of range"))?;
        int_literal(value, suffix, span)
    }

    fn punct(&mut self) -> Result<Tok, Error> {
        let span = self.span();
        let c0 = self.peek(0).unwrap_or('\0');
        let c1 = self.peek(1).unwrap_or('\0');
        let c2 = self.peek(2).unwrap_or('\0');
        let (tok, len) = match (c0, c1, c2) {
            ('<', '<', '=') => (Tok::ShlEq, 3),
            ('>', '>', '=') => (Tok::ShrEq, 3),
            ('-', '>', _) => (Tok::Arrow, 2),
            ('+', '+', _) => (Tok::PlusPlus, 2),
            ('-', '-', _) => (Tok::MinusMinus, 2),
            ('+', '=', _) => (Tok::PlusEq, 2),
            ('-', '=', _) => (Tok::MinusEq, 2),
            ('*', '=', _) => (Tok::StarEq, 2),
            ('/', '=', _) => (Tok::SlashEq, 2),
            ('%', '=', _) => (Tok::PercentEq, 2),
            ('&', '=', _) => (Tok::AmpEq, 2),
            ('|', '=', _) => (Tok::PipeEq, 2),
            ('^', '=', _) => (Tok::CaretEq, 2),
            ('<', '<', _) => (Tok::Shl, 2),
            ('>', '>', _) => (Tok::Shr, 2),
            ('<', '=', _) => (Tok::Le, 2),
            ('>', '=', _) => (Tok::Ge, 2),
            ('=', '=', _) => (Tok::EqEq, 2),
            ('!', '=', _) => (Tok::NotEq, 2),
            ('&', '&', _) => (Tok::AmpAmp, 2),
            ('|', '|', _) => (Tok::PipePipe, 2),
            ('(', _, _) => (Tok::LParen, 1),
            (')', _, _) => (Tok::RParen, 1),
            ('{', _, _) => (Tok::LBrace, 1),
            ('}', _, _) => (Tok::RBrace, 1),
            ('[', _, _) => (Tok::LBracket, 1),
            (']', _, _) => (Tok::RBracket, 1),
            (';', _, _) => (Tok::Semi, 1),
            (':', _, _) => (Tok::Colon, 1),
            (',', _, _) => (Tok::Comma, 1),
            ('.', _, _) => (Tok::Dot, 1),
            ('@', _, _) => (Tok::At, 1),
            ('=', _, _) => (Tok::Assign, 1),
            ('<', _, _) => (Tok::Lt, 1),
            ('>', _, _) => (Tok::Gt, 1),
            ('+', _, _) => (Tok::Plus, 1),
            ('-', _, _) => (Tok::Minus, 1),
            ('*', _, _) => (Tok::Star, 1),
            ('/', _, _) => (Tok::Slash, 1),
            ('%', _, _) => (Tok::Percent, 1),
            ('&', _, _) => (Tok::Amp, 1),
            ('|', _, _) => (Tok::Pipe, 1),
            ('^', _, _) => (Tok::Caret, 1),
            ('~', _, _) => (Tok::Tilde, 1),
            ('!', _, _) => (Tok::Bang, 1),
            _ => {
                let shown = if c0.is_control() {
                    format!("U+{:04X}", c0 as u32)
                } else {
                    c0.to_string()
                };
                return Err(span.error(format!("unexpected character '{shown}'")));
            }
        };
        for _ in 0..len {
            self.bump();
        }
        Ok(tok)
    }
}

fn is_blank(c: char) -> bool {
    matches!(
        c,
        ' ' | '\t'
            | '\n'
            | '\r'
            | '\u{0B}'
            | '\u{0C}'
            | '\u{85}'
            | '\u{200E}'
            | '\u{200F}'
            | '\u{2028}'
            | '\u{2029}'
    )
}

fn int_literal(value: u64, suffix: Option<char>, span: Span) -> Result<Tok, Error> {
    match suffix {
        Some('i') => i32::try_from(value)
            .map(|v| Tok::Int(IntLit::I32(v)))
            .map_err(|_| span.error(format!("integer literal {value}i does not fit in i32"))),
        Some('u') => u32::try_from(value)
            .map(|v| Tok::Int(IntLit::U32(v)))
            .map_err(|_| span.error(format!("integer literal {value}u does not fit in u32"))),
        _ => i64::try_from(value)
            .map(|v| Tok::Int(IntLit::Abstract(v)))
            .map_err(|_| {
                span.error("integer literal is out of range for a 64-bit abstract integer")
            }),
    }
}

/// WGSL template-list discovery (WGSL §3.9), run over tokens instead of code points.
/// `>>`, `>=` and `>>=` are split when their first `>` closes a template list.
fn discover_templates(toks: &mut Vec<Token>) {
    let mut pending: Vec<(usize, u32)> = Vec::new();
    let mut depth: u32 = 0;
    let mut i = 0;
    while i < toks.len() {
        match toks[i].tok {
            Tok::Ident(_) => {
                if matches!(toks.get(i + 1).map(|t| &t.tok), Some(Tok::Lt)) {
                    pending.push((i + 1, depth));
                    i += 2;
                } else {
                    i += 1;
                }
            }
            Tok::Gt | Tok::Shr | Tok::Ge | Tok::ShrEq => {
                if let Some(&(start, d)) = pending.last()
                    && d == depth
                {
                    pending.pop();
                    toks[start].tok = Tok::TemplateStart;
                    let rest = match toks[i].tok {
                        Tok::Shr => Some(Tok::Gt),
                        Tok::Ge => Some(Tok::Assign),
                        Tok::ShrEq => Some(Tok::Ge),
                        _ => None,
                    };
                    toks[i].tok = Tok::TemplateEnd;
                    if let Some(rest) = rest {
                        let span = Span {
                            line: toks[i].span.line,
                            col: toks[i].span.col + 1,
                        };
                        toks.insert(i + 1, Token { tok: rest, span });
                    }
                }
                i += 1;
            }
            Tok::LParen | Tok::LBracket => {
                depth += 1;
                i += 1;
            }
            Tok::RParen | Tok::RBracket => {
                while pending.last().is_some_and(|&(_, d)| d >= depth) {
                    pending.pop();
                }
                depth = depth.saturating_sub(1);
                i += 1;
            }
            Tok::Assign
            | Tok::PlusEq
            | Tok::MinusEq
            | Tok::StarEq
            | Tok::SlashEq
            | Tok::PercentEq
            | Tok::AmpEq
            | Tok::PipeEq
            | Tok::CaretEq
            | Tok::ShlEq
            | Tok::Semi
            | Tok::LBrace
            | Tok::Colon => {
                depth = 0;
                pending.clear();
                i += 1;
            }
            Tok::AmpAmp | Tok::PipePipe => {
                while pending.last().is_some_and(|&(_, d)| d >= depth) {
                    pending.pop();
                }
                i += 1;
            }
            _ => i += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(src: &str) -> Vec<Tok> {
        tokenize(src).unwrap().into_iter().map(|t| t.tok).collect()
    }

    #[test]
    fn integer_literals() {
        assert_eq!(toks("12")[0], Tok::Int(IntLit::Abstract(12)));
        assert_eq!(toks("0")[0], Tok::Int(IntLit::Abstract(0)));
        assert_eq!(toks("0xFFu")[0], Tok::Int(IntLit::U32(255)));
        assert_eq!(toks("0x45d9f3bu")[0], Tok::Int(IntLit::U32(0x45d9f3b)));
        assert_eq!(toks("7i")[0], Tok::Int(IntLit::I32(7)));
        assert_eq!(toks("4294967295u")[0], Tok::Int(IntLit::U32(u32::MAX)));
        assert_eq!(toks("0x1f")[0], Tok::Int(IntLit::Abstract(31)));
        assert!(tokenize("4294967296u").is_err());
        assert!(tokenize("2147483648i").is_err());
        assert!(tokenize("012").is_err());
        assert!(tokenize("12abc").is_err());
    }

    #[test]
    fn float_literals() {
        assert_eq!(toks("1.0")[0], Tok::Float(FloatLit::Abstract(1.0)));
        assert_eq!(toks(".5")[0], Tok::Float(FloatLit::Abstract(0.5)));
        assert_eq!(toks("1e-3")[0], Tok::Float(FloatLit::Abstract(1e-3)));
        assert_eq!(toks("2.")[0], Tok::Float(FloatLit::Abstract(2.0)));
        assert_eq!(toks("1.5e2")[0], Tok::Float(FloatLit::Abstract(150.0)));
        assert_eq!(toks("1f")[0], Tok::Float(FloatLit::F32(1.0)));
        assert_eq!(toks("1.0f")[0], Tok::Float(FloatLit::F32(1.0)));
        assert_eq!(toks("0f")[0], Tok::Float(FloatLit::F32(0.0)));
        assert_eq!(toks("0x1.8p1")[0], Tok::Float(FloatLit::Abstract(3.0)));
        assert_eq!(toks("0x1p-2f")[0], Tok::Float(FloatLit::F32(0.25)));
        assert!(tokenize("1.0h").is_err());
        assert!(tokenize("1e").is_err());
        assert!(tokenize("1.0u").is_err());
    }

    #[test]
    fn comments_and_positions() {
        let t = tokenize("a /* x /* nested */ y */ b // tail\n  c").unwrap();
        assert_eq!(t[0].tok, Tok::Ident("a".into()));
        assert_eq!(t[1].tok, Tok::Ident("b".into()));
        assert_eq!(t[2].tok, Tok::Ident("c".into()));
        assert_eq!(t[2].span, Span { line: 2, col: 3 });
        assert!(tokenize("a /* /* */").is_err());
    }

    #[test]
    fn operators() {
        assert_eq!(
            toks("a <<= b >>= c -> d ++ -- != && ||"),
            vec![
                Tok::Ident("a".into()),
                Tok::ShlEq,
                Tok::Ident("b".into()),
                Tok::ShrEq,
                Tok::Ident("c".into()),
                Tok::Arrow,
                Tok::Ident("d".into()),
                Tok::PlusPlus,
                Tok::MinusMinus,
                Tok::NotEq,
                Tok::AmpAmp,
                Tok::PipePipe,
                Tok::Eof
            ]
        );
    }

    #[test]
    fn template_discovery() {
        let t = toks("array<vec2<f32>>");
        assert_eq!(
            t,
            vec![
                Tok::Ident("array".into()),
                Tok::TemplateStart,
                Tok::Ident("vec2".into()),
                Tok::TemplateStart,
                Tok::Ident("f32".into()),
                Tok::TemplateEnd,
                Tok::TemplateEnd,
                Tok::Eof
            ]
        );
        let t = toks("a < b && c > d");
        assert!(t.contains(&Tok::Lt) && t.contains(&Tok::Gt));
        let t = toks("for (var i = 0; i < 4; i++) {}");
        assert!(t.contains(&Tok::Lt));
        let t = toks("select(a, b, x < y)");
        assert!(t.contains(&Tok::Lt));
        let t = toks("var x: vec2<f32>= y;");
        assert!(t.contains(&Tok::TemplateEnd) && t.contains(&Tok::Assign));
        let t = toks("x = a << b;");
        assert!(t.contains(&Tok::Shl));
    }
}
