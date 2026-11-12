//! Abstract syntax tree produced by the parser. Every node carries the span of its first token
//! (binary expressions carry the span of their operator).

use crate::lexer::{FloatLit, IntLit, Span};

#[derive(Clone, Debug)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Attr {
    pub name: String,
    pub args: Vec<Expr>,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
    AddrOf,
    Deref,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Shl,
    Shr,
    And,
    Or,
    Xor,
    LogicalAnd,
    LogicalOr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl BinaryOp {
    pub fn symbol(self) -> &'static str {
        match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Rem => "%",
            BinaryOp::Shl => "<<",
            BinaryOp::Shr => ">>",
            BinaryOp::And => "&",
            BinaryOp::Or => "|",
            BinaryOp::Xor => "^",
            BinaryOp::LogicalAnd => "&&",
            BinaryOp::LogicalOr => "||",
            BinaryOp::Eq => "==",
            BinaryOp::Ne => "!=",
            BinaryOp::Lt => "<",
            BinaryOp::Le => "<=",
            BinaryOp::Gt => ">",
            BinaryOp::Ge => ">=",
        }
    }

    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge
        )
    }
}

#[derive(Clone, Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum ExprKind {
    /// An identifier, possibly template-elaborated (`vec3<f32>`, `array<T, N>`); also used for types.
    Ident {
        name: String,
        templates: Option<Vec<Expr>>,
    },
    Int(IntLit),
    Float(FloatLit),
    Bool(bool),
    Unary(UnaryOp, Box<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    Call {
        name: String,
        templates: Option<Vec<Expr>>,
        args: Vec<Expr>,
    },
    Member(Box<Expr>, Ident),
    Index(Box<Expr>, Box<Expr>),
}

#[derive(Clone, Debug)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum StmtKind {
    Empty,
    Let {
        name: Ident,
        ty: Option<Expr>,
        init: Expr,
    },
    Var {
        name: Ident,
        space: Option<Vec<Expr>>,
        ty: Option<Expr>,
        init: Option<Expr>,
    },
    Const {
        name: Ident,
        ty: Option<Expr>,
        init: Expr,
    },
    Assign {
        lhs: Expr,
        op: Option<BinaryOp>,
        rhs: Expr,
    },
    Phony(Expr),
    Increment(Expr),
    Decrement(Expr),
    Call(Expr),
    If {
        cond: Expr,
        then: Block,
        otherwise: Option<Box<Stmt>>,
    },
    Loop {
        body: Block,
        continuing: Option<Continuing>,
    },
    For {
        init: Option<Box<Stmt>>,
        cond: Option<Expr>,
        update: Option<Box<Stmt>>,
        body: Block,
    },
    While {
        cond: Expr,
        body: Block,
    },
    Break,
    Continue,
    Discard,
    Return(Option<Expr>),
    Block(Block),
    ConstAssert(Expr),
}

#[derive(Clone, Debug)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    /// Span of the closing brace.
    pub end: Span,
}

#[derive(Clone, Debug)]
pub struct Continuing {
    pub body: Block,
    pub break_if: Option<Expr>,
}

#[derive(Clone, Debug)]
pub struct MemberDecl {
    pub attrs: Vec<Attr>,
    pub name: Ident,
    pub ty: Expr,
}

#[derive(Clone, Debug)]
pub struct StructDecl {
    pub name: Ident,
    pub members: Vec<MemberDecl>,
}

#[derive(Clone, Debug)]
pub struct GlobalVarDecl {
    pub attrs: Vec<Attr>,
    pub name: Ident,
    pub space: Option<Vec<Expr>>,
    pub ty: Option<Expr>,
    pub init: Option<Expr>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct ParamDecl {
    pub attrs: Vec<Attr>,
    pub name: Ident,
    pub ty: Expr,
}

#[derive(Clone, Debug)]
pub struct FnDecl {
    pub attrs: Vec<Attr>,
    pub name: Ident,
    pub params: Vec<ParamDecl>,
    pub ret: Option<(Vec<Attr>, Expr)>,
    pub body: Block,
}

#[derive(Clone, Debug)]
pub enum Decl {
    Struct(StructDecl),
    Alias {
        name: Ident,
        ty: Expr,
    },
    Const {
        name: Ident,
        ty: Option<Expr>,
        init: Expr,
    },
    Var(GlobalVarDecl),
    Fn(FnDecl),
    ConstAssert(Expr),
}

impl Decl {
    pub fn name(&self) -> Option<&Ident> {
        match self {
            Decl::Struct(s) => Some(&s.name),
            Decl::Alias { name, .. } | Decl::Const { name, .. } => Some(name),
            Decl::Var(v) => Some(&v.name),
            Decl::Fn(f) => Some(&f.name),
            Decl::ConstAssert(_) => None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Module {
    pub decls: Vec<Decl>,
}
