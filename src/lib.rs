#![doc = include_str!("readme.md")]
#![feature(try_trait_v2)]
#![no_std]

pub mod fold;
pub mod foldable;

pub mod prelude {
    pub use core::convert::Infallible;
    pub use core::ops::ControlFlow::{Break, Continue};
    pub use core::ops::{ControlFlow, Try};

    pub use crate::fold::Fold;
    pub use crate::foldable::{Foldable, IntoFoldable};
}
