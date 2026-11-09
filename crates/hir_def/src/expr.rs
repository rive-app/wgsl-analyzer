use la_arena::Idx;
pub use syntax::ast::operators::*;
use syntax::ast::{self, IncrDecr};

use crate::{
    body::BindingId,
    db::Interned,
    module_data::Name,
    type_ref::{AccessMode, StorageClass, TypeRef, VecDimensionality},
};

pub type ExprId = Idx<Expr>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Literal {
    Int(i64, BuiltinInt),
    Uint(u64, BuiltinUint),
    Float(u32, BuiltinFloat), // FIXME: f32 is not Eq
    Bool(bool),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BuiltinFloat {
    F32,
    Abstract,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BuiltinInt {
    I32,
    Abstract,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BuiltinUint {
    U32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Callee {
    InferredComponentMatrix {
        rows: VecDimensionality,
        columns: VecDimensionality,
    },
    InferredComponentVec(VecDimensionality),
    InferredComponentArray,
    Name(Name),
    Type(Interned<TypeRef>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Missing,
    BinaryOp {
        lhs: ExprId,
        rhs: ExprId,
        op: BinaryOp,
    },
    UnaryOp {
        expr: ExprId,
        op: UnaryOp,
    },
    Field {
        expr: ExprId,
        name: Name,
    },
    Call {
        callee: Callee,
        args: Vec<ExprId>,
    },
    Index {
        lhs: ExprId,
        index: ExprId,
    },
    Bitcast {
        expr: ExprId,
        ty: Interned<TypeRef>,
    },
    Literal(Literal),
    Path(Name),
}

pub type StatementId = Idx<Statement>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Statement {
    Missing,
    Compound {
        statements: Vec<StatementId>,
    },
    LetStatement {
        binding_id: BindingId,
        type_ref: Option<Interned<TypeRef>>,
        initializer: Option<ExprId>,
    },
    ConstStatement {
        binding_id: BindingId,
        type_ref: Option<Interned<TypeRef>>,
        initializer: Option<ExprId>,
    },
    VariableStatement {
        binding_id: BindingId,
        type_ref: Option<Interned<TypeRef>>,
        initializer: Option<ExprId>,
        storage_class: Option<StorageClass>,
        access_mode: Option<AccessMode>,
    },
    Return {
        expr: Option<ExprId>,
    },
    Assignment {
        lhs: ExprId,
        rhs: ExprId,
    },
    CompoundAssignment {
        lhs: ExprId,
        rhs: ExprId,
        op: CompoundOp,
    },
    IncrDecr {
        expr: ExprId,
        op: IncrDecr,
    },
    If {
        condition: ExprId,
        block: StatementId,
        else_if_blocks: Vec<StatementId>,
        else_block: Option<StatementId>,
    },
    For {
        initializer: Option<StatementId>,
        condition: Option<ExprId>,
        continuing_part: Option<StatementId>,
        block: StatementId,
    },
    While {
        condition: ExprId,
        block: StatementId,
    },
    Switch {
        expr: ExprId,
        case_blocks: Vec<(Vec<ExprId>, StatementId)>,
        default_block: Option<StatementId>,
    },
    Loop {
        body: StatementId,
    },
    Discard,
    Break,
    Continue,
    Continuing {
        block: StatementId,
    },
    // only function calls are allowed in this position. TODO add diagnostic
    Expr {
        expr: ExprId,
    },
}

fn hex_digits(text: &str) -> Option<&str> {
    text.strip_prefix("0x").or_else(|| text.strip_prefix("0X"))
}

fn float_kind(suffixed: bool) -> BuiltinFloat {
    if suffixed {
        BuiltinFloat::F32
    } else {
        BuiltinFloat::Abstract
    }
}

pub fn parse_literal(lit: ast::LiteralKind) -> Literal {
    match lit {
        ast::LiteralKind::IntLiteral(lit) => {
            let kind = if lit.text().ends_with('i') {
                BuiltinInt::I32
            } else {
                BuiltinInt::Abstract
            };
            let text = lit.text().trim_end_matches('i');
            let (text, negative) = match text.strip_prefix('-') {
                Some(new) => (new, true),
                None => (text, false),
            };
            // Out of range literals are left for naga to report.
            let mut value = match hex_digits(text) {
                Some(hex) => i64::from_str_radix(hex, 16),
                None => text.parse(),
            }
            .unwrap_or(0);

            if negative {
                value = -value;
            }

            Literal::Int(value, kind)
        },
        ast::LiteralKind::UintLiteral(lit) => {
            let text = lit.text().trim_end_matches('u');
            let value = match hex_digits(text) {
                Some(hex) => u64::from_str_radix(hex, 16),
                None => text.parse(),
            }
            .unwrap_or(0);

            Literal::Uint(value, BuiltinUint::U32)
        },
        // A hex float's `f` is a digit unless it follows the exponent.
        ast::LiteralKind::HexFloatLiteral(lit) => {
            let text = lit.text();
            let suffixed = text.contains(['p', 'P']) && text.ends_with(['f', 'h']);
            Literal::Float(0, float_kind(suffixed))
        },
        ast::LiteralKind::DecimalFloatLiteral(lit) => {
            Literal::Float(0, float_kind(lit.text().ends_with(['f', 'h'])))
        },
        ast::LiteralKind::True(_) => Literal::Bool(true),
        ast::LiteralKind::False(_) => Literal::Bool(false),
    }
}

impl Expr {
    pub fn walk_child_exprs(
        &self,
        mut f: impl FnMut(ExprId),
    ) {
        match self {
            Expr::BinaryOp { lhs, rhs, .. } => {
                f(*lhs);
                f(*rhs);
            },
            Expr::UnaryOp { expr, .. } => {
                f(*expr);
            },
            Expr::Field { expr, .. } => {
                f(*expr);
            },
            Expr::Call { args, .. } => {
                args.iter().copied().for_each(f);
            },
            Expr::Index { lhs, index } => {
                f(*lhs);
                f(*index);
            },
            Expr::Bitcast { expr, .. } => {
                f(*expr);
            },
            Expr::Missing => {},
            Expr::Literal(_) => {},
            Expr::Path(_) => {},
        }
    }
}
