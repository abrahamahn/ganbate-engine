//! Recursive-descent parser following the WGSL grammar, including its operator rules:
//! shifts, comparisons and bitwise operators do not chain or mix without parentheses.

use crate::Error;
use crate::ast::*;
use crate::lexer::{Span, Tok, Token};

const KEYWORDS: &[&str] = &[
    "alias",
    "break",
    "case",
    "const",
    "const_assert",
    "continue",
    "continuing",
    "default",
    "diagnostic",
    "discard",
    "else",
    "enable",
    "false",
    "fn",
    "for",
    "if",
    "let",
    "loop",
    "override",
    "requires",
    "return",
    "struct",
    "switch",
    "true",
    "var",
    "while",
];

/// Maximum syntactic nesting (blocks, `else if` chains, parentheses, unary operators); deeper
/// input is rejected instead of overflowing the stack in the recursive later passes.
pub(crate) const MAX_DEPTH: u32 = 128;

/// Maximum depth of an expression tree, which also bounds left-deep operator chains such as
/// `a + b + c + ...`.
pub(crate) const MAX_EXPR_DEPTH: usize = 128;

/// Depth of an expression tree, computed without recursion.
fn expr_depth(e: &Expr) -> usize {
    let mut max = 0;
    let mut stack = vec![(e, 1usize)];
    while let Some((e, d)) = stack.pop() {
        max = max.max(d);
        match &e.kind {
            ExprKind::Ident { templates, .. } => {
                for t in templates.iter().flatten() {
                    stack.push((t, d + 1));
                }
            }
            ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) => {}
            ExprKind::Unary(_, x) | ExprKind::Member(x, _) => stack.push((x, d + 1)),
            ExprKind::Binary(_, l, r) | ExprKind::Index(l, r) => {
                stack.push((l, d + 1));
                stack.push((r, d + 1));
            }
            ExprKind::Call {
                templates, args, ..
            } => {
                for a in templates.iter().flatten().chain(args) {
                    stack.push((a, d + 1));
                }
            }
        }
    }
    max
}

pub fn parse(toks: Vec<Token>) -> Result<Module, Error> {
    let mut p = Parser {
        toks,
        pos: 0,
        depth: 0,
        in_expr: false,
    };
    p.module()
}

struct Parser {
    toks: Vec<Token>,
    pos: usize,
    depth: u32,
    /// Inside an expression (so nested `expression()` calls are not outermost).
    in_expr: bool,
}

type R<T> = Result<T, Error>;

impl Parser {
    fn peek(&self) -> &Tok {
        &self.toks[self.pos.min(self.toks.len() - 1)].tok
    }

    fn peek_at(&self, n: usize) -> &Tok {
        &self.toks[(self.pos + n).min(self.toks.len() - 1)].tok
    }

    fn span(&self) -> Span {
        self.toks[self.pos.min(self.toks.len() - 1)].span
    }

    fn bump(&mut self) -> Token {
        let t = self.toks[self.pos.min(self.toks.len() - 1)].clone();
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == t {
            self.bump();
            true
        } else {
            false
        }
    }

    fn is_kw(&self, kw: &str) -> bool {
        matches!(self.peek(), Tok::Ident(s) if s == kw)
    }

    fn eat_kw(&mut self, kw: &str) -> bool {
        if self.is_kw(kw) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn unexpected(&self, expected: &str) -> Error {
        self.span().error(format!(
            "expected {expected}, found {}",
            self.peek().describe()
        ))
    }

    fn expect(&mut self, t: Tok) -> R<Span> {
        if self.peek() == &t {
            Ok(self.bump().span)
        } else {
            Err(self.unexpected(&format!("'{}'", t.text())))
        }
    }

    fn ident(&mut self, what: &str) -> R<Ident> {
        match self.peek().clone() {
            Tok::Ident(name) => {
                let span = self.span();
                if KEYWORDS.contains(&name.as_str()) {
                    return Err(
                        span.error(format!("expected {what}, found reserved word '{name}'"))
                    );
                }
                self.bump();
                Ok(Ident { name, span })
            }
            _ => Err(self.unexpected(what)),
        }
    }

    fn enter(&mut self) -> R<()> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.span().error("expression or block nesting is too deep"));
        }
        Ok(())
    }

    fn leave(&mut self) {
        self.depth -= 1;
    }

    // ---------------------------------------------------------------- module scope

    fn module(&mut self) -> R<Module> {
        let mut decls = Vec::new();
        loop {
            match self.peek().clone() {
                Tok::Eof => break,
                Tok::Semi => {
                    self.bump();
                }
                Tok::Ident(kw) if kw == "enable" || kw == "requires" => {
                    let span = self.span();
                    self.bump();
                    let what = match self.peek() {
                        Tok::Ident(s) => s.clone(),
                        _ => String::new(),
                    };
                    return Err(span.error(format!(
                        "'{kw} {what}' is not supported: Ganbate implements core WGSL only (no f16 or other extensions)"
                    )));
                }
                Tok::Ident(kw) if kw == "diagnostic" => {
                    // Diagnostic directives only tune uniformity diagnostics, which Ganbate does not emit.
                    while !matches!(self.peek(), Tok::Semi | Tok::Eof) {
                        self.bump();
                    }
                    self.expect(Tok::Semi)?;
                }
                _ => decls.push(self.global_decl()?),
            }
        }
        Ok(Module { decls })
    }

    fn global_decl(&mut self) -> R<Decl> {
        let attrs = self.attributes()?;
        let span = self.span();
        let kw = match self.peek() {
            Tok::Ident(s) => s.clone(),
            _ => return Err(self.unexpected("a declaration")),
        };
        if !attrs.is_empty() && !matches!(kw.as_str(), "fn" | "var" | "override") {
            return Err(attrs[0]
                .span
                .error(format!("attributes are not allowed on '{kw}' declarations")));
        }
        match kw.as_str() {
            "fn" => Ok(Decl::Fn(self.fn_decl(attrs)?)),
            "var" => {
                self.bump();
                let space = self.var_template()?;
                let name = self.ident("a variable name")?;
                let ty = if self.eat(&Tok::Colon) {
                    Some(self.type_expr()?)
                } else {
                    None
                };
                let init = if self.eat(&Tok::Assign) {
                    Some(self.expression()?)
                } else {
                    None
                };
                self.expect(Tok::Semi)?;
                Ok(Decl::Var(GlobalVarDecl {
                    attrs,
                    name,
                    space,
                    ty,
                    init,
                    span,
                }))
            }
            "const" => {
                self.bump();
                let (name, ty, init) = self.const_rest()?;
                self.expect(Tok::Semi)?;
                Ok(Decl::Const { name, ty, init })
            }
            "override" => {
                Err(span.error("'override' declarations are not supported by Ganbate; use 'const'"))
            }
            "struct" => {
                self.bump();
                let name = self.ident("a struct name")?;
                self.expect(Tok::LBrace)?;
                let mut members = Vec::new();
                while !matches!(self.peek(), Tok::RBrace) {
                    let attrs = self.attributes()?;
                    let mname = self.ident("a struct member name")?;
                    self.expect(Tok::Colon)?;
                    let ty = self.type_expr()?;
                    members.push(MemberDecl {
                        attrs,
                        name: mname,
                        ty,
                    });
                    if !self.eat(&Tok::Comma) {
                        if matches!(self.peek(), Tok::Semi) {
                            return Err(self
                                .span()
                                .error("struct members are separated by ',' in WGSL, not ';'"));
                        }
                        break;
                    }
                }
                self.expect(Tok::RBrace)?;
                Ok(Decl::Struct(StructDecl { name, members }))
            }
            "alias" => {
                self.bump();
                let name = self.ident("an alias name")?;
                self.expect(Tok::Assign)?;
                let ty = self.type_expr()?;
                self.expect(Tok::Semi)?;
                Ok(Decl::Alias { name, ty })
            }
            "const_assert" => {
                self.bump();
                let e = self.expression()?;
                self.expect(Tok::Semi)?;
                Ok(Decl::ConstAssert(e))
            }
            "let" => Err(span.error("'let' is not allowed at module scope; use 'const'")),
            _ => Err(self.unexpected("a declaration ('fn', 'var', 'const', 'struct' or 'alias')")),
        }
    }

    fn var_template(&mut self) -> R<Option<Vec<Expr>>> {
        if matches!(self.peek(), Tok::TemplateStart) {
            Ok(Some(self.template_args()?))
        } else if matches!(self.peek(), Tok::Lt) {
            Err(self
                .span()
                .error("malformed address space: expected 'var<space>'"))
        } else {
            Ok(None)
        }
    }

    fn const_rest(&mut self) -> R<(Ident, Option<Expr>, Expr)> {
        let name = self.ident("a constant name")?;
        let ty = if self.eat(&Tok::Colon) {
            Some(self.type_expr()?)
        } else {
            None
        };
        if !matches!(self.peek(), Tok::Assign) {
            return Err(self
                .span()
                .error("a 'const' declaration requires an initializer"));
        }
        self.bump();
        let init = self.expression()?;
        Ok((name, ty, init))
    }

    fn fn_decl(&mut self, attrs: Vec<Attr>) -> R<FnDecl> {
        self.bump(); // fn
        let name = self.ident("a function name")?;
        self.expect(Tok::LParen)?;
        let mut params = Vec::new();
        while !matches!(self.peek(), Tok::RParen) {
            let pattrs = self.attributes()?;
            let pname = self.ident("a parameter name")?;
            self.expect(Tok::Colon)?;
            let ty = self.type_expr()?;
            params.push(ParamDecl {
                attrs: pattrs,
                name: pname,
                ty,
            });
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(Tok::RParen)?;
        let ret = if self.eat(&Tok::Arrow) {
            let rattrs = self.attributes()?;
            Some((rattrs, self.type_expr()?))
        } else {
            None
        };
        let body = self.block()?;
        Ok(FnDecl {
            attrs,
            name,
            params,
            ret,
            body,
        })
    }

    fn attributes(&mut self) -> R<Vec<Attr>> {
        let mut attrs = Vec::new();
        while matches!(self.peek(), Tok::At) {
            let span = self.bump().span;
            let name = match self.bump().tok {
                Tok::Ident(s) => s,
                other => {
                    return Err(span.error(format!(
                        "expected an attribute name after '@', found {}",
                        other.describe()
                    )));
                }
            };
            let mut args = Vec::new();
            if self.eat(&Tok::LParen) {
                while !matches!(self.peek(), Tok::RParen) {
                    args.push(self.expression()?);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.expect(Tok::RParen)?;
            }
            attrs.push(Attr { name, args, span });
        }
        Ok(attrs)
    }

    /// A type: a (possibly template-elaborated) identifier.
    fn type_expr(&mut self) -> R<Expr> {
        let span = self.span();
        match self.peek().clone() {
            Tok::Ident(name) => {
                if KEYWORDS.contains(&name.as_str()) {
                    return Err(
                        span.error(format!("expected a type, found reserved word '{name}'"))
                    );
                }
                self.bump();
                let templates = if matches!(self.peek(), Tok::TemplateStart) {
                    Some(self.template_args()?)
                } else {
                    None
                };
                Ok(Expr {
                    kind: ExprKind::Ident { name, templates },
                    span,
                })
            }
            _ => Err(self.unexpected("a type")),
        }
    }

    fn template_args(&mut self) -> R<Vec<Expr>> {
        self.expect(Tok::TemplateStart)?;
        let mut args = Vec::new();
        while !matches!(self.peek(), Tok::TemplateEnd) {
            args.push(self.expression()?);
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        if !matches!(self.peek(), Tok::TemplateEnd) {
            return Err(self.unexpected("'>' to close the template list"));
        }
        self.bump();
        Ok(args)
    }

    // ---------------------------------------------------------------- statements

    fn block(&mut self) -> R<Block> {
        self.expect(Tok::LBrace)?;
        self.enter()?;
        let mut stmts = Vec::new();
        while !matches!(self.peek(), Tok::RBrace) {
            if matches!(self.peek(), Tok::Eof) {
                return Err(self.unexpected("'}'"));
            }
            if self.is_kw("continuing") {
                return Err(self
                    .span()
                    .error("'continuing' is only allowed as the last statement of a 'loop' body"));
            }
            stmts.push(self.statement()?);
        }
        let end = self.bump().span;
        self.leave();
        Ok(Block { stmts, end })
    }

    fn statement(&mut self) -> R<Stmt> {
        let span = self.span();
        let kind = match self.peek().clone() {
            Tok::Semi => {
                self.bump();
                StmtKind::Empty
            }
            Tok::LBrace => StmtKind::Block(self.block()?),
            Tok::Ident(kw) => match kw.as_str() {
                "return" => {
                    self.bump();
                    let value = if matches!(self.peek(), Tok::Semi) {
                        None
                    } else {
                        Some(self.expression()?)
                    };
                    self.expect(Tok::Semi)?;
                    StmtKind::Return(value)
                }
                "if" => self.if_stmt()?,
                "loop" => self.loop_stmt()?,
                "for" => self.for_stmt()?,
                "while" => {
                    self.bump();
                    let cond = self.expression()?;
                    let body = self.block()?;
                    StmtKind::While { cond, body }
                }
                "break" => {
                    self.bump();
                    if self.is_kw("if") {
                        return Err(span.error("'break if' is only allowed as the last statement of a 'continuing' block"));
                    }
                    self.expect(Tok::Semi)?;
                    StmtKind::Break
                }
                "continue" => {
                    self.bump();
                    self.expect(Tok::Semi)?;
                    StmtKind::Continue
                }
                "discard" => {
                    self.bump();
                    self.expect(Tok::Semi)?;
                    StmtKind::Discard
                }
                "switch" => {
                    return Err(span.error(
                        "'switch' statements are not supported by Ganbate; use if/else chains",
                    ));
                }
                "const_assert" => {
                    self.bump();
                    let e = self.expression()?;
                    self.expect(Tok::Semi)?;
                    StmtKind::ConstAssert(e)
                }
                "else" => return Err(span.error("'else' without a matching 'if'")),
                _ => {
                    let k = self.simple_stmt()?;
                    self.expect(Tok::Semi)?;
                    k
                }
            },
            _ => {
                let k = self.simple_stmt()?;
                self.expect(Tok::Semi)?;
                k
            }
        };
        Ok(Stmt { kind, span })
    }

    /// Declarations, assignments, increments and call statements (no trailing ';').
    fn simple_stmt(&mut self) -> R<StmtKind> {
        if self.eat_kw("let") {
            let name = self.ident("a name")?;
            let ty = if self.eat(&Tok::Colon) {
                Some(self.type_expr()?)
            } else {
                None
            };
            if !matches!(self.peek(), Tok::Assign) {
                return Err(self
                    .span()
                    .error("a 'let' declaration requires an initializer"));
            }
            self.bump();
            let init = self.expression()?;
            return Ok(StmtKind::Let { name, ty, init });
        }
        if self.eat_kw("var") {
            let space = self.var_template()?;
            let name = self.ident("a variable name")?;
            let ty = if self.eat(&Tok::Colon) {
                Some(self.type_expr()?)
            } else {
                None
            };
            let init = if self.eat(&Tok::Assign) {
                Some(self.expression()?)
            } else {
                None
            };
            return Ok(StmtKind::Var {
                name,
                space,
                ty,
                init,
            });
        }
        if self.eat_kw("const") {
            let (name, ty, init) = self.const_rest()?;
            return Ok(StmtKind::Const { name, ty, init });
        }
        if matches!(self.peek(), Tok::Underscore) {
            self.bump();
            self.expect(Tok::Assign)?;
            return Ok(StmtKind::Phony(self.expression()?));
        }
        let lhs = self.unary()?;
        let op = match self.peek() {
            Tok::Assign => None,
            Tok::PlusEq => Some(BinaryOp::Add),
            Tok::MinusEq => Some(BinaryOp::Sub),
            Tok::StarEq => Some(BinaryOp::Mul),
            Tok::SlashEq => Some(BinaryOp::Div),
            Tok::PercentEq => Some(BinaryOp::Rem),
            Tok::AmpEq => Some(BinaryOp::And),
            Tok::PipeEq => Some(BinaryOp::Or),
            Tok::CaretEq => Some(BinaryOp::Xor),
            Tok::ShlEq => Some(BinaryOp::Shl),
            Tok::ShrEq => Some(BinaryOp::Shr),
            Tok::PlusPlus => {
                self.bump();
                return Ok(StmtKind::Increment(lhs));
            }
            Tok::MinusMinus => {
                self.bump();
                return Ok(StmtKind::Decrement(lhs));
            }
            _ => {
                if matches!(lhs.kind, ExprKind::Call { .. }) {
                    return Ok(StmtKind::Call(lhs));
                }
                return Err(self.unexpected("an assignment, increment, or function call statement"));
            }
        };
        self.bump();
        let rhs = self.expression()?;
        Ok(StmtKind::Assign { lhs, op, rhs })
    }

    fn if_stmt(&mut self) -> R<StmtKind> {
        self.bump(); // if
        let cond = self.expression()?;
        let then = self.block()?;
        let otherwise = if self.is_kw("else") {
            self.bump();
            let span = self.span();
            if self.is_kw("if") {
                self.enter()?;
                let k = self.if_stmt()?;
                self.leave();
                Some(Box::new(Stmt { kind: k, span }))
            } else {
                Some(Box::new(Stmt {
                    kind: StmtKind::Block(self.block()?),
                    span,
                }))
            }
        } else {
            None
        };
        Ok(StmtKind::If {
            cond,
            then,
            otherwise,
        })
    }

    fn loop_stmt(&mut self) -> R<StmtKind> {
        self.bump(); // loop
        self.expect(Tok::LBrace)?;
        self.enter()?;
        let mut stmts = Vec::new();
        let mut continuing = None;
        loop {
            match self.peek() {
                Tok::RBrace => break,
                Tok::Eof => return Err(self.unexpected("'}'")),
                _ => {}
            }
            if self.is_kw("continuing") {
                self.bump();
                self.expect(Tok::LBrace)?;
                self.enter()?;
                let mut cstmts = Vec::new();
                let mut break_if = None;
                while !matches!(self.peek(), Tok::RBrace) {
                    if matches!(self.peek(), Tok::Eof) {
                        return Err(self.unexpected("'}'"));
                    }
                    if self.is_kw("break") && matches!(self.peek_at(1), Tok::Ident(s) if s == "if")
                    {
                        self.bump();
                        self.bump();
                        break_if = Some(self.expression()?);
                        self.expect(Tok::Semi)?;
                        if !matches!(self.peek(), Tok::RBrace) {
                            return Err(self.span().error(
                                "'break if' must be the last statement of a 'continuing' block",
                            ));
                        }
                        break;
                    }
                    if self.is_kw("continuing") {
                        return Err(self.span().error("nested 'continuing' block"));
                    }
                    cstmts.push(self.statement()?);
                }
                let cend = self.expect(Tok::RBrace)?;
                self.leave();
                continuing = Some(Continuing {
                    body: Block {
                        stmts: cstmts,
                        end: cend,
                    },
                    break_if,
                });
                if !matches!(self.peek(), Tok::RBrace) {
                    return Err(self
                        .span()
                        .error("'continuing' must be the last statement of a 'loop' body"));
                }
                break;
            }
            stmts.push(self.statement()?);
        }
        let end = self.expect(Tok::RBrace)?;
        self.leave();
        Ok(StmtKind::Loop {
            body: Block { stmts, end },
            continuing,
        })
    }

    fn for_stmt(&mut self) -> R<StmtKind> {
        self.bump(); // for
        self.expect(Tok::LParen)?;
        let init = if matches!(self.peek(), Tok::Semi) {
            None
        } else {
            let span = self.span();
            Some(Box::new(Stmt {
                kind: self.simple_stmt()?,
                span,
            }))
        };
        self.expect(Tok::Semi)?;
        let cond = if matches!(self.peek(), Tok::Semi) {
            None
        } else {
            Some(self.expression()?)
        };
        self.expect(Tok::Semi)?;
        let update = if matches!(self.peek(), Tok::RParen) {
            None
        } else {
            let span = self.span();
            let kind = self.simple_stmt()?;
            if matches!(
                kind,
                StmtKind::Let { .. } | StmtKind::Var { .. } | StmtKind::Const { .. }
            ) {
                return Err(span.error("a 'for' update cannot be a declaration"));
            }
            Some(Box::new(Stmt { kind, span }))
        };
        self.expect(Tok::RParen)?;
        let body = self.block()?;
        Ok(StmtKind::For {
            init,
            cond,
            update,
            body,
        })
    }

    // ---------------------------------------------------------------- expressions

    fn binary_op(tok: &Tok) -> Option<BinaryOp> {
        Some(match tok {
            Tok::Plus => BinaryOp::Add,
            Tok::Minus => BinaryOp::Sub,
            Tok::Star => BinaryOp::Mul,
            Tok::Slash => BinaryOp::Div,
            Tok::Percent => BinaryOp::Rem,
            Tok::Shl => BinaryOp::Shl,
            Tok::Shr => BinaryOp::Shr,
            Tok::Amp => BinaryOp::And,
            Tok::Pipe => BinaryOp::Or,
            Tok::Caret => BinaryOp::Xor,
            Tok::AmpAmp => BinaryOp::LogicalAnd,
            Tok::PipePipe => BinaryOp::LogicalOr,
            Tok::EqEq => BinaryOp::Eq,
            Tok::NotEq => BinaryOp::Ne,
            Tok::Lt => BinaryOp::Lt,
            Tok::Le => BinaryOp::Le,
            Tok::Gt => BinaryOp::Gt,
            Tok::Ge => BinaryOp::Ge,
            _ => return None,
        })
    }

    fn bin(op: BinaryOp, l: Expr, r: Expr, span: Span) -> Expr {
        Expr {
            kind: ExprKind::Binary(op, Box::new(l), Box::new(r)),
            span,
        }
    }

    fn ambiguous(&self, prev: BinaryOp) -> Error {
        let next = Self::binary_op(self.peek()).map_or("?", |o| o.symbol());
        self.span().error(format!(
            "'{}' cannot follow '{}' without parentheses (WGSL does not define a precedence between them)",
            next,
            prev.symbol()
        ))
    }

    pub fn expression(&mut self) -> R<Expr> {
        let outermost = !self.in_expr;
        let saved = self.in_expr;
        self.in_expr = true;
        self.enter()?;
        let r = self.expression_inner();
        self.leave();
        self.in_expr = saved;
        let e = r?;
        if outermost && expr_depth(&e) > MAX_EXPR_DEPTH {
            return Err(e.span.error(format!(
                "expression is too deeply nested (more than {MAX_EXPR_DEPTH} levels); split it with 'let'"
            )));
        }
        Ok(e)
    }

    fn expression_inner(&mut self) -> R<Expr> {
        let lhs = self.unary()?;
        if let Some(op @ (BinaryOp::And | BinaryOp::Or | BinaryOp::Xor)) =
            Self::binary_op(self.peek())
        {
            let mut e = lhs;
            while Self::binary_op(self.peek()) == Some(op) {
                let span = self.bump().span;
                let r = self.unary()?;
                e = Self::bin(op, e, r, span);
            }
            if Self::binary_op(self.peek()).is_some() {
                return Err(self.ambiguous(op));
            }
            return Ok(e);
        }
        let rel = self.relational_rest(lhs)?;
        if let Some(op @ (BinaryOp::LogicalAnd | BinaryOp::LogicalOr)) =
            Self::binary_op(self.peek())
        {
            let mut e = rel;
            while Self::binary_op(self.peek()) == Some(op) {
                let span = self.bump().span;
                let r = self.relational()?;
                e = Self::bin(op, e, r, span);
            }
            if Self::binary_op(self.peek()).is_some() {
                return Err(self.ambiguous(op));
            }
            return Ok(e);
        }
        if let Some(op) = Self::binary_op(self.peek()) {
            return Err(self.span().error(format!(
                "operator '{}' cannot be used here without parentheses",
                op.symbol()
            )));
        }
        Ok(rel)
    }

    fn relational(&mut self) -> R<Expr> {
        let u = self.unary()?;
        self.relational_rest(u)
    }

    fn relational_rest(&mut self, lhs: Expr) -> R<Expr> {
        let l = self.shift_rest(lhs)?;
        if let Some(op) = Self::binary_op(self.peek())
            && op.is_comparison()
        {
            let span = self.bump().span;
            let r = self.shift()?;
            let e = Self::bin(op, l, r, span);
            if let Some(next) = Self::binary_op(self.peek())
                && next.is_comparison()
            {
                return Err(self.ambiguous(op));
            }
            return Ok(e);
        }
        Ok(l)
    }

    fn shift(&mut self) -> R<Expr> {
        let u = self.unary()?;
        self.shift_rest(u)
    }

    fn shift_rest(&mut self, lhs: Expr) -> R<Expr> {
        if let Some(op @ (BinaryOp::Shl | BinaryOp::Shr)) = Self::binary_op(self.peek()) {
            let span = self.bump().span;
            let r = self.unary()?;
            let e = Self::bin(op, lhs, r, span);
            if let Some(
                BinaryOp::Shl
                | BinaryOp::Shr
                | BinaryOp::Add
                | BinaryOp::Sub
                | BinaryOp::Mul
                | BinaryOp::Div
                | BinaryOp::Rem,
            ) = Self::binary_op(self.peek())
            {
                return Err(self.ambiguous(op));
            }
            return Ok(e);
        }
        let m = self.multiplicative_rest(lhs)?;
        let a = self.additive_rest(m)?;
        if let Some(BinaryOp::Shl | BinaryOp::Shr) = Self::binary_op(self.peek()) {
            return Err(self.span().error(
                "a shift cannot follow an arithmetic expression without parentheses (WGSL shift operands must be unary expressions)",
            ));
        }
        Ok(a)
    }

    fn multiplicative_rest(&mut self, mut lhs: Expr) -> R<Expr> {
        while let Some(op @ (BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem)) =
            Self::binary_op(self.peek())
        {
            let span = self.bump().span;
            let r = self.unary()?;
            lhs = Self::bin(op, lhs, r, span);
        }
        Ok(lhs)
    }

    fn additive_rest(&mut self, mut lhs: Expr) -> R<Expr> {
        while let Some(op @ (BinaryOp::Add | BinaryOp::Sub)) = Self::binary_op(self.peek()) {
            let span = self.bump().span;
            let u = self.unary()?;
            let r = self.multiplicative_rest(u)?;
            lhs = Self::bin(op, lhs, r, span);
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> R<Expr> {
        let span = self.span();
        let op = match self.peek() {
            Tok::Minus => Some(UnaryOp::Neg),
            Tok::Bang => Some(UnaryOp::Not),
            Tok::Tilde => Some(UnaryOp::BitNot),
            Tok::Amp => Some(UnaryOp::AddrOf),
            Tok::Star => Some(UnaryOp::Deref),
            Tok::MinusMinus => {
                return Err(span.error("'--' cannot be used as a prefix operator; write '-(-x)'"));
            }
            Tok::PlusPlus => return Err(span.error("'++' cannot be used as a prefix operator")),
            _ => None,
        };
        if let Some(op) = op {
            self.bump();
            self.enter()?;
            let inner = self.unary();
            self.leave();
            return Ok(Expr {
                kind: ExprKind::Unary(op, Box::new(inner?)),
                span,
            });
        }
        let mut e = self.primary()?;
        loop {
            match self.peek() {
                Tok::Dot => {
                    self.bump();
                    let span = self.span();
                    let name = match self.peek().clone() {
                        Tok::Ident(s) => s,
                        _ => return Err(self.unexpected("a member name after '.'")),
                    };
                    self.bump();
                    let espan = e.span;
                    e = Expr {
                        kind: ExprKind::Member(Box::new(e), Ident { name, span }),
                        span: espan,
                    };
                }
                Tok::LBracket => {
                    self.bump();
                    let idx = self.expression()?;
                    self.expect(Tok::RBracket)?;
                    let espan = e.span;
                    e = Expr {
                        kind: ExprKind::Index(Box::new(e), Box::new(idx)),
                        span: espan,
                    };
                }
                _ => break,
            }
        }
        Ok(e)
    }

    fn primary(&mut self) -> R<Expr> {
        let span = self.span();
        let kind = match self.peek().clone() {
            Tok::Int(v) => {
                self.bump();
                ExprKind::Int(v)
            }
            Tok::Float(v) => {
                self.bump();
                ExprKind::Float(v)
            }
            Tok::LParen => {
                self.bump();
                let e = self.expression()?;
                self.expect(Tok::RParen)?;
                return Ok(e);
            }
            Tok::Ident(name) => {
                match name.as_str() {
                    "true" => {
                        self.bump();
                        return Ok(Expr {
                            kind: ExprKind::Bool(true),
                            span,
                        });
                    }
                    "false" => {
                        self.bump();
                        return Ok(Expr {
                            kind: ExprKind::Bool(false),
                            span,
                        });
                    }
                    _ => {}
                }
                if KEYWORDS.contains(&name.as_str()) {
                    return Err(span.error(format!(
                        "expected an expression, found reserved word '{name}'"
                    )));
                }
                self.bump();
                let templates = if matches!(self.peek(), Tok::TemplateStart) {
                    Some(self.template_args()?)
                } else {
                    None
                };
                if matches!(self.peek(), Tok::LParen) {
                    self.bump();
                    let mut args = Vec::new();
                    while !matches!(self.peek(), Tok::RParen) {
                        args.push(self.expression()?);
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                    }
                    self.expect(Tok::RParen)?;
                    ExprKind::Call {
                        name,
                        templates,
                        args,
                    }
                } else {
                    ExprKind::Ident { name, templates }
                }
            }
            _ => return Err(self.unexpected("an expression")),
        };
        Ok(Expr { kind, span })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;

    fn expr(src: &str) -> Result<Expr, Error> {
        let mut p = Parser {
            toks: tokenize(src).unwrap(),
            pos: 0,
            depth: 0,
            in_expr: false,
        };
        let e = p.expression()?;
        if !matches!(p.peek(), Tok::Eof) {
            return Err(p.unexpected("end of expression"));
        }
        Ok(e)
    }

    fn show(e: &Expr) -> String {
        match &e.kind {
            ExprKind::Ident { name, templates } => match templates {
                Some(t) => format!(
                    "{name}<{}>",
                    t.iter().map(show).collect::<Vec<_>>().join(",")
                ),
                None => name.clone(),
            },
            ExprKind::Int(v) => format!("{v:?}"),
            ExprKind::Float(v) => format!("{v:?}"),
            ExprKind::Bool(b) => b.to_string(),
            ExprKind::Unary(op, x) => format!("({op:?} {})", show(x)),
            ExprKind::Binary(op, l, r) => format!("({} {} {})", show(l), op.symbol(), show(r)),
            ExprKind::Call {
                name,
                templates,
                args,
            } => format!(
                "{name}{}({})",
                templates
                    .as_ref()
                    .map(|t| format!("<{}>", t.iter().map(show).collect::<Vec<_>>().join(",")))
                    .unwrap_or_default(),
                args.iter().map(show).collect::<Vec<_>>().join(",")
            ),
            ExprKind::Member(b, m) => format!("{}.{}", show(b), m.name),
            ExprKind::Index(b, i) => format!("{}[{}]", show(b), show(i)),
        }
    }

    #[test]
    fn precedence() {
        assert_eq!(show(&expr("a + b * c").unwrap()), "(a + (b * c))");
        assert_eq!(show(&expr("a * b + c").unwrap()), "((a * b) + c)");
        assert_eq!(show(&expr("a - b - c").unwrap()), "((a - b) - c)");
        assert_eq!(
            show(&expr("a < b + 1 && c").unwrap()).replace("Abstract(1)", "1"),
            "((a < (b + 1)) && c)"
        );
        assert_eq!(show(&expr("x ^ (x >> y)").unwrap()), "(x ^ (x >> y))");
        assert_eq!(show(&expr("-a.b[c]").unwrap()), "(Neg a.b[c])");
        assert_eq!(show(&expr("a & b & c").unwrap()), "((a & b) & c)");
        assert_eq!(show(&expr("a || b || c").unwrap()), "((a || b) || c)");
        assert_eq!(show(&expr("!a == b").unwrap()), "((Not a) == b)");
    }

    #[test]
    fn wgsl_rejects_ambiguous_mixes() {
        assert!(expr("a & b | c").is_err());
        assert!(expr("a < b < c").is_err());
        assert!(expr("a && b || c").is_err());
        assert!(expr("a + b << c").is_err());
        assert!(expr("a << b + c").is_err());
        assert!(expr("a ^ b >> c").is_err());
        assert!(expr("a & b == c").is_err());
    }

    #[test]
    fn templates_and_calls() {
        assert_eq!(
            show(&expr("array<vec2<f32>, 4>(a, b)").unwrap()).replace("Abstract(4)", "4"),
            "array<vec2<f32>,4>(a,b)"
        );
        assert_eq!(show(&expr("bitcast<f32>(x)").unwrap()), "bitcast<f32>(x)");
        assert_eq!(
            show(&expr("select(a, b, x < y)").unwrap()),
            "select(a,b,(x < y))"
        );
        assert_eq!(
            show(&expr("a < b && c > d").unwrap()),
            "((a < b) && (c > d))"
        );
    }

    #[test]
    fn statements_parse() {
        let src = "fn f() { var i = 0; loop { i++; if (i > 3) { break; } continuing { i += 1; break if i >= 10; } } for (var j = 0u; j < 4u; j++) { continue; } while i < 3 { i--; } _ = i; }";
        let m = parse(tokenize(src).unwrap()).unwrap();
        assert_eq!(m.decls.len(), 1);
        assert!(parse(tokenize("fn f() { switch x { } }").unwrap()).is_err());
        assert!(parse(tokenize("fn f() { loop { continuing { } i++; } }").unwrap()).is_err());
        assert!(parse(tokenize("struct S { a: f32; b: f32 }").unwrap()).is_err());
    }

    #[test]
    fn deep_nesting_is_an_error_not_a_crash() {
        let src = format!("const x = {}1{};", "(".repeat(5000), ")".repeat(5000));
        assert!(parse(tokenize(&src).unwrap()).is_err());
        let src = format!("fn f() {{ {} }}", "{".repeat(5000));
        assert!(parse(tokenize(&src).unwrap()).is_err());
    }
}
