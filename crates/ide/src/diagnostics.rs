use base_db::{FileRange, TextRange};
use hir::{
    HirDatabase, Semantics,
    diagnostics::{AnyDiagnostic, DiagnosticsConfig},
};
use hir_def::original_file_range;
use hir_ty::ty::{
    self, Ty, VecSize,
    pretty::{pretty_fn, pretty_type},
};
use itertools::Itertools;
use rowan::NodeOrToken;
use syntax::AstNode;
use vfs::FileId;

pub struct Diagnostic {
    pub code: DiagnosticCode,
    pub message: String,
    pub range: TextRange,
    pub unused: bool,
    pub severity: Severity,
    pub related: Vec<(String, FileRange)>,
}

pub struct DiagnosticCode(&'static str);

impl DiagnosticCode {
    pub fn url(&self) -> String {
        self.0.to_string()
    }

    pub fn as_str(&self) -> &'static str {
        self.0
    }
}

#[derive(Clone, Copy)]
pub enum Severity {
    Error,
    WeakWarning,
}

impl Diagnostic {
    pub fn new(
        code: DiagnosticCode,
        message: String,
        range: TextRange,
    ) -> Self {
        Self {
            code,
            message,
            range,
            severity: Severity::Error,
            unused: false,
            related: Vec::new(),
        }
    }

    pub fn with_severity(
        self,
        severity: Severity,
    ) -> Self {
        Diagnostic { severity, ..self }
    }

    pub fn unused(self) -> Self {
        Diagnostic {
            unused: true,
            ..self
        }
    }
}

pub fn diagnostics(
    db: &dyn HirDatabase,
    config: &DiagnosticsConfig,
    file_id: FileId,
) -> Vec<Diagnostic> {
    let (parse, unconfigured) = db.parse_with_unconfigured(file_id);

    let mut diagnostics = Vec::new();

    diagnostics.extend(
        parse
            .errors()
            .iter()
            .map(|error| AnyDiagnostic::ParseError {
                message: error.message(),
                range: error.range,
                file_id: file_id.into(),
            }),
    );

    diagnostics.extend(
        unconfigured
            .iter()
            .map(|unconfigured| AnyDiagnostic::UnconfiguredCode {
                def: unconfigured.def.clone(),
                range: unconfigured.range,
                file_id: file_id.into(),
            }),
    );

    let sema = Semantics::new(db);

    if config.type_errors {
        sema.module(file_id)
            .diagnostics(db, config, &mut diagnostics);
    }

    diagnostics
        .into_iter()
        .map(|diagnostic| {
            let file_id = diagnostic.file_id();
            let root = db.parse_or_resolve(file_id).unwrap().syntax();
            match diagnostic {
                AnyDiagnostic::AssignmentNotAReference { lhs, actual } => {
                    let source = lhs.value.to_node(&root);
                    let actual = ty::pretty::pretty_type(db, actual);
                    let frange = original_file_range(db.upcast(), lhs.file_id, source.syntax());
                    Diagnostic::new(
                        DiagnosticCode("1"),
                        format!(
                            "left hand side of assignment should be a reference, found {}",
                            actual
                        ),
                        frange.range,
                    )
                },
                AnyDiagnostic::TypeMismatch {
                    expr,
                    expected,
                    actual,
                } => {
                    let source = expr.value.to_node(&root);
                    let mut expected_text =
                        ty::pretty::pretty_type_expectation(db, expected.clone());
                    let mut actual_text = ty::pretty::pretty_type(db, actual);
                    // Compact types leave out address spaces and access modes.
                    if expected_text == actual_text {
                        expected_text = ty::pretty::pretty_type_expectation_with_verbosity(
                            db,
                            expected,
                            ty::pretty::TypeVerbosity::Full,
                        );
                        actual_text = ty::pretty::pretty_type_with_verbosity(
                            db,
                            actual,
                            ty::pretty::TypeVerbosity::Full,
                        );
                    }
                    let frange = original_file_range(db.upcast(), expr.file_id, source.syntax());
                    Diagnostic::new(
                        DiagnosticCode("2"),
                        format!("expected {}, found {}", expected_text, actual_text),
                        frange.range,
                    )
                },
                AnyDiagnostic::NoSuchField { expr, name, ty } => {
                    let source = expr.value.to_node(&root).syntax().parent().unwrap();
                    let ty = ty::pretty::pretty_type(db, ty);
                    let frange = original_file_range(db.upcast(), expr.file_id, source.syntax());
                    Diagnostic::new(
                        DiagnosticCode("3"),
                        format!("no field `{}` on type {}", name.as_ref(), ty),
                        frange.range,
                    )
                },
                AnyDiagnostic::ArrayAccessInvalidType { expr, ty } => {
                    let source = expr.value.to_node(&root);
                    let ty = ty::pretty::pretty_type(db, ty);
                    let frange = original_file_range(db.upcast(), expr.file_id, source.syntax());
                    Diagnostic::new(
                        DiagnosticCode("4"),
                        format!("cannot index into type {}", ty),
                        frange.range,
                    )
                },
                AnyDiagnostic::UnresolvedName { expr, name } => {
                    let source = expr.value.to_node(&root);
                    let frange = original_file_range(db.upcast(), expr.file_id, source.syntax());
                    Diagnostic::new(
                        DiagnosticCode("5"),
                        format!("cannot find `{}` in this scope", name.as_str()),
                        frange.range,
                    )
                },
                AnyDiagnostic::InvalidConstructionType { expr, ty } => {
                    let source = expr.value.to_node(&root);
                    let ty = ty::pretty::pretty_type(db, ty);
                    let frange = original_file_range(db.upcast(), expr.file_id, source.syntax());
                    Diagnostic::new(
                        DiagnosticCode("6"),
                        format!("cannot construct value of type {}", ty),
                        frange.range,
                    )
                },
                AnyDiagnostic::FunctionCallArgCountMismatch {
                    expr,
                    n_expected,
                    n_actual,
                } => {
                    let source = expr.value.to_node(&root).syntax().parent().unwrap();
                    let frange = original_file_range(db.upcast(), expr.file_id, source.syntax());
                    Diagnostic::new(
                        DiagnosticCode("7"),
                        format!("expected {} parameters, found {}", n_expected, n_actual),
                        frange.range,
                    )
                },
                AnyDiagnostic::NoBuiltinOverload {
                    expr,
                    builtin,
                    parameters,
                    name,
                } => {
                    let source = expr.value.to_node(&root).syntax().parent().unwrap();
                    let builtin = builtin.lookup(db);

                    let parameters = parameters
                        .iter()
                        .map(|ty| ty::pretty::pretty_type(db, *ty))
                        .join(", ");

                    let possible = builtin.signatures(db).join("\n");

                    let name = match name {
                        Some(name) => name,
                        None => builtin.name(),
                    };

                    let frange = original_file_range(db.upcast(), expr.file_id, source.syntax());
                    Diagnostic::new(
                        DiagnosticCode("8"),
                        format!(
                            "no overload of `{}` found for given arguments. \
                        Found ({}), expected one of:\n{}",
                            name, parameters, possible
                        ),
                        frange.range,
                    )
                },
                AnyDiagnostic::AddrOfNotRef { expr, actual } => {
                    let source = expr.value.to_node(&root);
                    let ty = ty::pretty::pretty_type(db, actual);
                    let frange = original_file_range(db.upcast(), expr.file_id, source.syntax());
                    Diagnostic::new(
                        DiagnosticCode("9"),
                        format!("expected a reference, found {}", ty),
                        frange.range,
                    )
                },
                AnyDiagnostic::DerefNotPtr { expr, actual } => {
                    let source = expr.value.to_node(&root);
                    let ty = ty::pretty::pretty_type(db, actual);
                    let frange = original_file_range(db.upcast(), expr.file_id, source.syntax());
                    Diagnostic::new(
                        DiagnosticCode("10"),
                        format!("cannot dereference expression of type {}", ty),
                        frange.range,
                    )
                },
                AnyDiagnostic::MissingStorageClass { var } => {
                    let var_decl = var.value.to_node(&root);
                    let source = var_decl
                        .var_token()
                        .map(NodeOrToken::Token)
                        .unwrap_or_else(|| NodeOrToken::Node(var_decl.syntax()));

                    let frange = original_file_range(db.upcast(), var.file_id, &source);
                    Diagnostic::new(
                        DiagnosticCode("11"),
                        "missing storage class on global variable".to_string(),
                        frange.range,
                    )
                },
                AnyDiagnostic::InvalidStorageClass { var, error } => {
                    let var_decl = var.value.to_node(&root);
                    let source = var_decl
                        .var_token()
                        .map(NodeOrToken::Token)
                        .unwrap_or_else(|| NodeOrToken::Node(var_decl.syntax()));
                    let frange = original_file_range(db.upcast(), var.file_id, &source);
                    Diagnostic::new(DiagnosticCode("12"), format!("{}", error), frange.range)
                },
                AnyDiagnostic::InvalidType {
                    file_id: _,
                    location,
                    error,
                } => {
                    let source = location.to_node(&root);
                    let frange = original_file_range(db.upcast(), file_id, source.syntax());
                    Diagnostic::new(DiagnosticCode("13"), format!("{}", error), frange.range)
                },
                AnyDiagnostic::UnresolvedImport { import } => {
                    let source = import.value.to_node(&root);
                    let frange = original_file_range(db.upcast(), file_id, source.syntax());
                    Diagnostic::new(
                        DiagnosticCode("14"),
                        "unresolved import".to_string(),
                        frange.range,
                    )
                },
                AnyDiagnostic::ParseError { message, range, .. } => {
                    Diagnostic::new(DiagnosticCode("16"), message, range)
                },
                AnyDiagnostic::UnconfiguredCode { def, range, .. } => Diagnostic::new(
                    DiagnosticCode("17"),
                    format!(
                        "code is inactive due to `#ifdef` directives: `{}` is not enabled",
                        def
                    ),
                    range,
                )
                .with_severity(Severity::WeakWarning)
                .unused(),
                AnyDiagnostic::NoConstructor {
                    expr,
                    builtins: [specific, general],
                    ty,
                    parameters,
                } => {
                    let source = expr.value.to_node(&root).syntax().clone();

                    let parameters = parameters
                        .iter()
                        .map(|ty| ty::pretty::pretty_type(db, *ty))
                        .join(", ");

                    let mut possible = Vec::with_capacity(32);
                    let builtin_specific = specific.lookup(db);
                    possible.extend(
                        builtin_specific
                            .overloads()
                            .map(|(_, overload)| pretty_fn(db, &overload.ty.lookup(db))),
                    );
                    let builtin_general = general.lookup(db);
                    possible.extend(
                        builtin_general
                            .overloads()
                            .filter(|(_, overload)| {
                                let function = overload.ty.lookup(db);
                                if let Some(return_ty) = function.return_type {
                                    convert_compatible(db, ty, return_ty)
                                } else {
                                    true
                                }
                            })
                            .map(|(_, overload)| pretty_fn(db, &overload.ty.lookup(db))),
                    );

                    let possible = possible.join("\n");

                    let frange = original_file_range(db.upcast(), expr.file_id, source.syntax());
                    Diagnostic::new(
                        DiagnosticCode("18"),
                        format!(
                            "no overload of constructor `{}` found for given \
                            arguments. Found ({parameters}), expected one of:\n{possible}",
                            pretty_type(db, ty),
                        ),
                        frange.range,
                    )
                },
                AnyDiagnostic::PrecedenceParensRequired {
                    expr,
                    op,
                    sequence_permitted,
                } => {
                    let source = expr.value.to_node(&root);
                    let frange = original_file_range(db.upcast(), file_id, source.syntax());
                    let symbol = op.symbol();
                    let message = if sequence_permitted {
                        format!(
                            "{symbol} sequences may only have unary operands.
                            More complex operands must be this with parenthesized `()`",
                        )
                    } else {
                        format!(
                            "{symbol} expressions may only have unary operands.
                            More complex operands must be this with parenthesized `()`"
                        )
                    };
                    Diagnostic::new(DiagnosticCode("19"), message, frange.range)
                },
            }
        })
        .collect()
}

fn size_compatible(
    target: VecSize,
    overload: VecSize,
) -> bool {
    match overload {
        VecSize::Two | VecSize::Three | VecSize::Four => overload == target,
        VecSize::BoundVar(_) => true,
    }
}

fn convert_compatible(
    db: &dyn HirDatabase,
    target: Ty,
    overload: Ty,
) -> bool {
    let target_kind = target.kind(db);
    let overload_kind = overload.kind(db);
    match (target_kind, overload_kind) {
        (ty::TyKind::Vector(tg), ty::TyKind::Vector(ov)) => {
            size_compatible(tg.size, ov.size) && convert_compatible(db, tg.inner, ov.inner)
        },
        (ty::TyKind::Matrix(tg), ty::TyKind::Matrix(ov)) => {
            size_compatible(tg.columns, ov.columns)
                && size_compatible(tg.rows, ov.rows)
                && convert_compatible(db, tg.inner, ov.inner)
        },
        (ty::TyKind::Scalar(s1), ty::TyKind::Scalar(s2)) => s1 == s2,
        _ => false,
    }
}
