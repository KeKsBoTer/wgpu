//! `requires …;` extensions in WGSL.
//!
//! The focal point of this module is the [`LanguageExtension`] API.

use strum::VariantArray;

/// A language extension recognized by Naga, but not guaranteed to be present in all environments.
///
/// WGSL spec.: <https://www.w3.org/TR/WGSL/#language-extensions-sec>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum LanguageExtension {
    #[allow(unused)]
    Implemented(ImplementedLanguageExtension),
    Unimplemented(UnimplementedLanguageExtension),
}

impl LanguageExtension {
    const READONLY_AND_READWRITE_STORAGE_TEXTURES: &'static str =
        "readonly_and_readwrite_storage_textures";
    const PACKED4X8_INTEGER_DOT_PRODUCT: &'static str = "packed_4x8_integer_dot_product";
    const UNRESTRICTED_POINTER_PARAMETERS: &'static str = "unrestricted_pointer_parameters";
    const POINTER_COMPOSITE_ACCESS: &'static str = "pointer_composite_access";
    const FRAGMENT_SHADER_INTERLOCK: &'static str = "fragment_shader_interlock";

    /// Convert from a sentinel word in WGSL into its associated [`LanguageExtension`], if possible.
    pub fn from_ident(s: &str) -> Option<Self> {
        Some(match s {
            Self::READONLY_AND_READWRITE_STORAGE_TEXTURES => Self::Unimplemented(
                UnimplementedLanguageExtension::ReadOnlyAndReadWriteStorageTextures,
            ),
            Self::PACKED4X8_INTEGER_DOT_PRODUCT => {
                Self::Unimplemented(UnimplementedLanguageExtension::Packed4x8IntegerDotProduct)
            }
            Self::UNRESTRICTED_POINTER_PARAMETERS => {
                Self::Unimplemented(UnimplementedLanguageExtension::UnrestrictedPointerParameters)
            }
            Self::POINTER_COMPOSITE_ACCESS => {
                Self::Implemented(ImplementedLanguageExtension::PointerCompositeAccess)
            }
            Self::FRAGMENT_SHADER_INTERLOCK => {
                Self::Implemented(ImplementedLanguageExtension::FragmentShaderInterlock)
            }
            _ => return None,
        })
    }

    /// Maps this [`LanguageExtension`] into the sentinel word associated with it in WGSL.
    pub const fn to_ident(self) -> &'static str {
        match self {
            Self::Implemented(kind) => kind.to_ident(),
            Self::Unimplemented(kind) => match kind {
                UnimplementedLanguageExtension::ReadOnlyAndReadWriteStorageTextures => {
                    Self::READONLY_AND_READWRITE_STORAGE_TEXTURES
                }
                UnimplementedLanguageExtension::Packed4x8IntegerDotProduct => {
                    Self::PACKED4X8_INTEGER_DOT_PRODUCT
                }
                UnimplementedLanguageExtension::UnrestrictedPointerParameters => {
                    Self::UNRESTRICTED_POINTER_PARAMETERS
                }
            },
        }
    }
}

/// A variant of [`LanguageExtension::Implemented`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, VariantArray)]
pub enum ImplementedLanguageExtension {
    PointerCompositeAccess,
    FragmentShaderInterlock
}

impl ImplementedLanguageExtension {
    /// Returns slice of all variants of [`ImplementedLanguageExtension`].
    pub const fn all() -> &'static [Self] {
        Self::VARIANTS
    }

    /// Maps this [`ImplementedLanguageExtension`] into the sentinel word associated with it in WGSL.
    pub const fn to_ident(self) -> &'static str {
        match self {
            ImplementedLanguageExtension::PointerCompositeAccess => {
                LanguageExtension::POINTER_COMPOSITE_ACCESS
            }
            ImplementedLanguageExtension::FragmentShaderInterlock => {
                LanguageExtension::FRAGMENT_SHADER_INTERLOCK
            }
        }
    }
}

/// A variant of [`LanguageExtension::Unimplemented`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum UnimplementedLanguageExtension {
    ReadOnlyAndReadWriteStorageTextures,
    Packed4x8IntegerDotProduct,
    UnrestrictedPointerParameters,
}

impl UnimplementedLanguageExtension {
    pub(crate) const fn tracking_issue_num(self) -> u16 {
        match self {
            Self::ReadOnlyAndReadWriteStorageTextures => 6204,
            Self::Packed4x8IntegerDotProduct => 6445,
            Self::UnrestrictedPointerParameters => 5158,
        }
    }
}