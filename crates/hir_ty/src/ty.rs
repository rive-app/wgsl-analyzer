pub mod pretty;

use std::{borrow::Cow, fmt::Write, str::FromStr};

pub use hir_def::type_ref::{AccessMode, StorageClass};
use hir_def::{db::StructId, module_data::Name, type_ref};
use salsa::InternKey;

use crate::HirDatabase;

// TODO:
// [ ] nesting depth
// [ ] constructable
// [ ] io-shareable
// [ ] host-shareable
// [ ] storable

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub struct Ty {
    ty: salsa::InternId,
}

impl InternKey for Ty {
    fn from_intern_id(ty: salsa::InternId) -> Self {
        Ty { ty }
    }

    fn as_intern_id(&self) -> salsa::InternId {
        self.ty
    }
}

impl Ty {
    pub fn kind(
        self,
        db: &dyn HirDatabase,
    ) -> TyKind {
        db.lookup_intern_ty(self)
    }

    pub fn is_err(
        self,
        db: &dyn HirDatabase,
    ) -> bool {
        matches!(db.lookup_intern_ty(self), TyKind::Error)
    }

    /// `T` -> `T`, `vecN<T>` -> `T`
    #[must_use]
    pub fn this_or_vec_inner(
        self,
        db: &dyn HirDatabase,
    ) -> Ty {
        match self.kind(db) {
            TyKind::Vector(vec) => vec.inner,
            TyKind::Ref(r) => r.inner.this_or_vec_inner(db),
            _ => self,
        }
    }

    /// `ref<inner>` -> `inner`, `ptr<inner>` -> `ptr<inner>`
    #[must_use]
    pub fn unref(
        self,
        db: &dyn HirDatabase,
    ) -> Ty {
        match self.kind(db) {
            TyKind::Ref(r) => r.inner,
            _ => self,
        }
    }

    pub fn contains_struct(
        self,
        db: &dyn HirDatabase,
        r#struct: StructId,
    ) -> bool {
        self.kind(db).contains_struct(db, r#struct)
    }

    /// The scalar of a scalar, or of a vector, matrix or array's elements.
    pub fn scalar(
        self,
        db: &dyn HirDatabase,
    ) -> Option<ScalarType> {
        match self.kind(db) {
            TyKind::Scalar(scalar) => Some(scalar),
            TyKind::Vector(VectorType { inner, .. })
            | TyKind::Matrix(MatrixType { inner, .. })
            | TyKind::Array(ArrayType { inner, .. }) => inner.scalar(db),
            _ => None,
        }
    }

    /// This type with its scalar, if it has one, replaced by `map`.
    #[must_use]
    pub fn map_scalar(
        self,
        db: &dyn HirDatabase,
        map: fn(ScalarType) -> ScalarType,
    ) -> Ty {
        match self.kind(db) {
            TyKind::Scalar(scalar) => TyKind::Scalar(map(scalar)),
            TyKind::Vector(vector) => TyKind::Vector(VectorType {
                inner: vector.inner.map_scalar(db, map),
                ..vector
            }),
            TyKind::Matrix(matrix) => TyKind::Matrix(MatrixType {
                inner: matrix.inner.map_scalar(db, map),
                ..matrix
            }),
            TyKind::Array(array) => TyKind::Array(ArrayType {
                inner: array.inner.map_scalar(db, map),
                ..array
            }),
            _ => return self,
        }
        .intern(db)
    }

    /// The type an abstract value takes where nothing asks for another, as in `let x = 1`.
    #[must_use]
    pub fn concretize(
        self,
        db: &dyn HirDatabase,
    ) -> Ty {
        self.map_scalar(db, |scalar| match scalar {
            ScalarType::AbstractInt => ScalarType::I32,
            ScalarType::AbstractFloat => ScalarType::F32,
            scalar => scalar,
        })
    }

    /// This type with its numeric scalar made abstract.
    #[must_use]
    pub fn abstractify(
        self,
        db: &dyn HirDatabase,
    ) -> Ty {
        self.map_scalar(db, |scalar| match scalar {
            ScalarType::I32 | ScalarType::U32 => ScalarType::AbstractInt,
            ScalarType::F32 => ScalarType::AbstractFloat,
            scalar => scalar,
        })
    }

    pub fn is_abstract(
        self,
        db: &dyn HirDatabase,
    ) -> bool {
        self.concretize(db) != self
    }

    /// Whether this is an error type, or a reference or pointer to one.
    pub fn is_err_behind_refs(
        self,
        db: &dyn HirDatabase,
    ) -> bool {
        match self.kind(db) {
            TyKind::Error => true,
            TyKind::Ref(Ref { inner, .. }) | TyKind::Ptr(Ptr { inner, .. }) => {
                inner.is_err_behind_refs(db)
            },
            _ => false,
        }
    }

    /// WGSL's rank for converting a value of this type to `to`, lower preferred,
    /// or `None` when it cannot.
    pub fn conversion_rank(
        self,
        db: &dyn HirDatabase,
        to: Ty,
    ) -> Option<u32> {
        if self == to {
            return Some(0);
        }
        match (self.kind(db), to.kind(db)) {
            (TyKind::Scalar(from), TyKind::Scalar(to)) => from.conversion_rank(&to),
            (TyKind::Vector(from), TyKind::Vector(to)) if from.size == to.size => {
                from.inner.conversion_rank(db, to.inner)
            },
            (TyKind::Matrix(from), TyKind::Matrix(to))
                if from.columns == to.columns && from.rows == to.rows =>
            {
                from.inner.conversion_rank(db, to.inner)
            },
            (TyKind::Array(from), TyKind::Array(to))
                if from.size.matches(&to.size) && from.binding_array == to.binding_array =>
            {
                from.inner.conversion_rank(db, to.inner)
            },
            (TyKind::Ptr(from), TyKind::Ptr(to))
                if from.storage_class == to.storage_class && from.access_mode == to.access_mode =>
            {
                from.inner.conversion_rank(db, to.inner)
            },
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TyKind {
    Error,
    Scalar(ScalarType),
    Atomic(AtomicType),
    Vector(VectorType),
    Matrix(MatrixType),
    Struct(StructId),
    Array(ArrayType),
    Texture(TextureType),
    Sampler(SamplerType),
    Ref(Ref),
    Ptr(Ptr),
    BoundVar(BoundVar),
    StorageTypeOfTexelFormat(BoundVar), // e.g. rgba8unorm -> vec4<f32>
    BuiltinStruct(BuiltinStruct),
}

/// The result of `modf`, `frexp` or `atomicCompareExchangeWeak`, a struct WGSL
/// never lets a shader name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BuiltinStruct {
    pub kind: BuiltinStructKind,
    pub inner: Ty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinStructKind {
    Modf,
    Frexp,
    AtomicCompareExchange,
}

impl BuiltinStruct {
    pub fn fields(
        self,
        db: &dyn HirDatabase,
    ) -> [(&'static str, Ty); 2] {
        match self.kind {
            BuiltinStructKind::Modf => [("fract", self.inner), ("whole", self.inner)],
            BuiltinStructKind::Frexp => {
                let i32 = TyKind::Scalar(ScalarType::I32).intern(db);
                let exp = match self.inner.kind(db) {
                    TyKind::Vector(vector) => TyKind::Vector(VectorType {
                        size: vector.size,
                        inner: i32,
                    })
                    .intern(db),
                    _ => i32,
                };
                [("fract", self.inner), ("exp", exp)]
            },
            BuiltinStructKind::AtomicCompareExchange => [
                ("old_value", self.inner),
                ("exchanged", TyKind::Scalar(ScalarType::Bool).intern(db)),
            ],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoundVar {
    pub index: usize,
}

impl TyKind {
    pub fn bool() -> TyKind {
        TyKind::Scalar(ScalarType::Bool)
    }

    pub fn f32() -> TyKind {
        TyKind::Scalar(ScalarType::Bool)
    }

    pub fn i32() -> TyKind {
        TyKind::Scalar(ScalarType::Bool)
    }

    pub fn u32() -> TyKind {
        TyKind::Scalar(ScalarType::Bool)
    }

    pub fn vec_of() -> TyKind {
        TyKind::Scalar(ScalarType::Bool)
    }
}

impl TyKind {
    pub fn unref(
        &self,
        db: &dyn HirDatabase,
    ) -> Cow<'_, TyKind> {
        match self {
            TyKind::Ref(r) => Cow::Owned(r.inner.kind(db)),
            _ => Cow::Borrowed(self),
        }
    }

    pub fn is_numeric_scalar(&self) -> bool {
        match self {
            TyKind::Scalar(scalar) => scalar.is_numeric(),
            _ => false,
        }
    }

    pub fn intern(
        self,
        db: &dyn HirDatabase,
    ) -> Ty {
        db.intern_ty(self)
    }

    pub fn is_err(&self) -> bool {
        matches!(self, TyKind::Error)
    }

    pub fn is_plain(&self) -> bool {
        matches!(
            self,
            TyKind::Scalar(_)
                | TyKind::Vector(_)
                | TyKind::Matrix(_)
                | TyKind::Atomic(_)
                | TyKind::Array(_)
                | TyKind::Struct(_)
        )
    }

    pub fn is_constructable(&self) -> bool {
        matches!(
            self,
            TyKind::Scalar(_)
                | TyKind::Vector(_)
                | TyKind::Matrix(_)
                | TyKind::Array(ArrayType {
                    size: ArraySize::Const(_) | ArraySize::Named(_),
                    ..
                })
                | TyKind::Struct(_)
                | TyKind::BuiltinStruct(_)
        )
    }

    pub fn is_storable(&self) -> bool {
        matches!(
            self,
            TyKind::Scalar(_)
                | TyKind::Vector(_)
                | TyKind::Matrix(_)
                | TyKind::Atomic(_)
                | TyKind::Array(_)
                | TyKind::Struct(_)
                | TyKind::Texture(_)
                | TyKind::Sampler(_)
        )
    }

    pub fn is_io_shareable(
        &self,
        db: &dyn HirDatabase,
    ) -> bool {
        match self {
            TyKind::Scalar(_) => true,
            TyKind::Vector(vec) => vec.inner.kind(db).is_numeric_scalar(),
            TyKind::Struct(r#struct) => {
                db.field_types(*r#struct)
                    .iter()
                    .all(|(_, ty)| match ty.kind(db) {
                        TyKind::Scalar(_) => true,
                        TyKind::Vector(vec) if vec.inner.kind(db).is_numeric_scalar() => true,
                        _ => false,
                    })
            },
            _ => false,
        }
    }

    pub fn is_host_shareable(
        &self,
        db: &dyn HirDatabase,
    ) -> bool {
        match self {
            TyKind::Scalar(scalar) => scalar.is_numeric(),
            TyKind::Vector(vec) => vec.inner.kind(db).is_numeric_scalar(),
            TyKind::Matrix(_) | TyKind::Atomic(_) => true,
            TyKind::Array(array) => array.inner.kind(db).is_host_shareable(db),
            TyKind::Struct(r#struct) => db
                .field_types(*r#struct)
                .iter()
                .all(|(_, ty)| ty.kind(db).is_host_shareable(db)),
            _ => false,
        }
    }

    pub fn contains_runtime_sized_array(
        &self,
        db: &dyn HirDatabase,
    ) -> bool {
        match self {
            TyKind::Array(ArrayType {
                size: ArraySize::Dynamic,
                ..
            }) => true,
            TyKind::Struct(r#struct) => db
                .field_types(*r#struct)
                .iter()
                .any(|(_, ty)| ty.kind(db).contains_runtime_sized_array(db)),
            _ => false,
        }
    }

    pub fn contains_struct(
        &self,
        db: &dyn HirDatabase,
        r#struct: StructId,
    ) -> bool {
        match self {
            TyKind::Atomic(atomic) => atomic.inner.contains_struct(db, r#struct),
            TyKind::Struct(id) => {
                if *id == r#struct {
                    return true;
                }
                db.field_types(*id)
                    .values()
                    .any(|ty| ty.contains_struct(db, r#struct))
            },
            TyKind::Array(array) => array.inner.contains_struct(db, r#struct),
            TyKind::Ref(r) => r.inner.contains_struct(db, r#struct),
            TyKind::Ptr(ptr) => ptr.inner.contains_struct(db, r#struct),
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ScalarType {
    Bool,
    I32,
    U32,
    F32,
    /// The type of an unsuffixed integer literal, which converts to what its use needs.
    AbstractInt,
    /// The type of an unsuffixed float literal, which converts to what its use needs.
    AbstractFloat,
}

impl ScalarType {
    pub fn is_numeric(&self) -> bool {
        !matches!(self, ScalarType::Bool)
    }

    /// WGSL's rank for converting to `to`, lower preferred, or `None` when it cannot.
    fn conversion_rank(
        &self,
        to: &ScalarType,
    ) -> Option<u32> {
        use ScalarType::*;
        match (self, to) {
            _ if self == to => Some(0),
            (AbstractFloat, F32) => Some(1),
            (AbstractInt, I32) => Some(3),
            (AbstractInt, U32) => Some(4),
            (AbstractInt, AbstractFloat) => Some(5),
            (AbstractInt, F32) => Some(6),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VecSize {
    Two,
    Three,
    Four,
    BoundVar(BoundVar),
}

impl TryFrom<u8> for VecSize {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Ok(match value {
            2 => VecSize::Two,
            3 => VecSize::Three,
            4 => VecSize::Four,
            _ => return Err(()),
        })
    }
}

impl From<type_ref::VecDimensionality> for VecSize {
    fn from(dim: type_ref::VecDimensionality) -> Self {
        match dim {
            type_ref::VecDimensionality::Two => VecSize::Two,
            type_ref::VecDimensionality::Three => VecSize::Three,
            type_ref::VecDimensionality::Four => VecSize::Four,
        }
    }
}

impl std::fmt::Display for VecSize {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            VecSize::Two => f.write_str("2"),
            VecSize::Three => f.write_str("3"),
            VecSize::Four => f.write_str("4"),
            VecSize::BoundVar(var) => {
                let mut names = "NMOPQRS".chars();
                write!(f, "{}", names.nth(var.index).unwrap())
            },
        }
    }
}

impl VecSize {
    pub fn as_u8(&self) -> u8 {
        match self {
            VecSize::Two => 2,
            VecSize::Three => 3,
            VecSize::Four => 4,
            VecSize::BoundVar(_) => panic!("VecSize::BoundVar cannot be made into an u8"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VectorType {
    pub size: VecSize,
    pub inner: Ty,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MatrixType {
    pub columns: VecSize,
    pub rows: VecSize,
    pub inner: Ty,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AtomicType {
    pub inner: Ty,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArrayType {
    pub inner: Ty,
    pub binding_array: bool,
    pub size: ArraySize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ArraySize {
    Const(u64),
    /// A length given by a constant, which we do not evaluate.
    Named(Name),
    Dynamic,
}

impl ArraySize {
    /// Whether arrays of these sizes can be the same type, as a named size could be any length.
    fn matches(
        &self,
        other: &ArraySize,
    ) -> bool {
        match (self, other) {
            (ArraySize::Dynamic, ArraySize::Dynamic) => true,
            (ArraySize::Dynamic, _) | (_, ArraySize::Dynamic) => false,
            (ArraySize::Named(_), _) | (_, ArraySize::Named(_)) => true,
            (ArraySize::Const(a), ArraySize::Const(b)) => a == b,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ptr {
    pub storage_class: StorageClass,
    pub inner: Ty,
    pub access_mode: AccessMode,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ref {
    pub storage_class: StorageClass,
    pub inner: Ty,
    pub access_mode: AccessMode,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TextureType {
    pub kind: TextureKind,
    pub dimension: TextureDimensionality,
    pub arrayed: bool,
    pub multisampled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TextureKind {
    Sampled(Ty),
    Storage(TexelFormat, AccessMode),
    Depth,
    External,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TextureDimensionality {
    D1,
    D2,
    D3,
    Cube,
}

impl std::fmt::Display for TextureDimensionality {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            TextureDimensionality::D1 => f.write_str("1d"),
            TextureDimensionality::D2 => f.write_str("2d"),
            TextureDimensionality::D3 => f.write_str("3d"),
            TextureDimensionality::Cube => f.write_str("cube"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SamplerType {
    pub comparison: bool,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum TexelFormat {
    Rgba8unorm,
    Rgba8snorm,
    Rgba8uint,
    Rgba8sint,
    Rgba16uint,
    Rgba16sint,
    Rgba16float,
    Rgba32uint,
    Rgba32sint,
    Rgba32float,

    R32uint,
    R32sint,
    R32float,
    Rg32uint,
    Rg32sint,
    Rg32float,

    BoundVar(BoundVar),
    // this is only used for builtins which do not care about the format
    Any,
}

impl std::fmt::Display for TexelFormat {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let str = match self {
            TexelFormat::Rgba8unorm => "rgba8unorm",
            TexelFormat::Rgba8snorm => "rgba8snorm",
            TexelFormat::Rgba8uint => "rgba8uint",
            TexelFormat::Rgba8sint => "rgba8sint",
            TexelFormat::Rgba16uint => "rgba16uint",
            TexelFormat::Rgba16sint => "rgba16sint",
            TexelFormat::Rgba16float => "rgba16float",
            TexelFormat::Rgba32uint => "rgba32uint",
            TexelFormat::Rgba32sint => "rgba32sint",
            TexelFormat::Rgba32float => "rgba32float",
            TexelFormat::R32uint => "r32uint",
            TexelFormat::R32sint => "r32sint",
            TexelFormat::R32float => "r32float",
            TexelFormat::Rg32uint => "rg32uint",
            TexelFormat::Rg32sint => "rg32sint",
            TexelFormat::Rg32float => "rg32float",
            TexelFormat::BoundVar(var) => return f.write_char(('F'..).nth(var.index).unwrap()),
            TexelFormat::Any => "_",
        };
        f.write_str(str)
    }
}

impl FromStr for TexelFormat {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "rgba8unorm" => TexelFormat::Rgba8unorm,
            "rgba8snorm" => TexelFormat::Rgba8snorm,
            "rgba8uint" => TexelFormat::Rgba8uint,
            "rgba8sint" => TexelFormat::Rgba8sint,
            "rgba16uint" => TexelFormat::Rgba16uint,
            "rgba16sint" => TexelFormat::Rgba16sint,
            "rgba16float" => TexelFormat::Rgba16float,
            "rgba32uint" => TexelFormat::Rgba32uint,
            "rgba32sint" => TexelFormat::Rgba32sint,
            "rgba32float" => TexelFormat::Rgba32float,
            "r32uint" => TexelFormat::R32uint,
            "r32sint" => TexelFormat::R32sint,
            "r32float" => TexelFormat::R32float,
            "rg32uint" => TexelFormat::Rg32uint,
            "rg32sint" => TexelFormat::Rg32sint,
            "rg32float" => TexelFormat::Rg32float,
            _ => return Err(()),
        })
    }
}
