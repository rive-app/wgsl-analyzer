use hir_def::module_data::Name;

use crate::{
    HirDatabase,
    function::{FunctionDetails, ResolvedFunctionId},
    ty::{pretty::pretty_fn_variants, *},
};

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub struct BuiltinId(salsa::InternId);
impl salsa::InternKey for BuiltinId {
    fn from_intern_id(id: salsa::InternId) -> Self {
        BuiltinId(id)
    }

    fn as_intern_id(&self) -> salsa::InternId {
        self.0
    }
}

impl BuiltinId {
    pub fn lookup(
        self,
        db: &dyn HirDatabase,
    ) -> Builtin {
        db.lookup_intern_builtin(self)
    }
}

impl Builtin {
    pub fn intern(
        self,
        db: &dyn HirDatabase,
    ) -> BuiltinId {
        db.intern_builtin(self)
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub enum GenericArgKind {
    VecSize,
    Type,
    TexelFormat,
}

pub enum GenericArg {
    VecSize(VecSize),
    Type(Ty),
    TexelFormat(TexelFormat),
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct Builtin {
    name: Name,
    overloads: Vec<BuiltinOverload>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BuiltinOverloadId(usize);

impl Builtin {
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    pub fn overloads(&self) -> impl Iterator<Item = (BuiltinOverloadId, &BuiltinOverload)> {
        self.overloads
            .iter()
            .enumerate()
            .map(|(i, overload)| (BuiltinOverloadId(i), overload))
    }

    pub fn overload(
        &self,
        overload_id: BuiltinOverloadId,
    ) -> &BuiltinOverload {
        &self.overloads[overload_id.0]
    }

    /// Each table line of this builtin, printed once for all the integer variants it expands to.
    pub fn signatures(
        &self,
        db: &dyn HirDatabase,
    ) -> Vec<String> {
        self.overloads
            .chunk_by(|a, b| a.line == b.line)
            .map(|variants| {
                let variants: Vec<_> = variants
                    .iter()
                    .map(|overload| overload.ty.lookup(db))
                    .collect();
                pretty_fn_variants(db, &variants)
            })
            .collect()
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct BuiltinOverload {
    /// The table line this overload was expanded from.
    pub line: usize,
    pub generics: Vec<GenericArgKind>,
    pub ty: ResolvedFunctionId,
}

include!(concat!(env!("OUT_DIR"), "/generated/builtins.rs"));
