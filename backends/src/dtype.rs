use std::fmt;

#[cfg(feature = "clap")]
use clap::ValueEnum;

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "clap", derive(Clone, ValueEnum))]
pub enum DType {
    // Float16 is not available on accelerate
    #[cfg(any(
        feature = "python",
        all(feature = "candle", not(feature = "accelerate"))
    ))]
    Float16,
    BFloat16,
    #[cfg(any(feature = "python", feature = "candle", feature = "ort"))]
    Float32,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            // Float16 is not available on accelerate
            #[cfg(any(
                feature = "python",
                all(feature = "candle", not(feature = "accelerate"))
            ))]
            DType::Float16 => write!(f, "float16"),
            DType::BFloat16 => write!(f, "bfloat16"),
            #[cfg(any(feature = "python", feature = "candle", feature = "ort"))]
            DType::Float32 => write!(f, "float32"),
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for DType {
    fn default() -> Self {
        #[cfg(any(
            feature = "accelerate",
            feature = "mkl",
            feature = "mkl-dynamic",
            feature = "ort"
        ))]
        {
            DType::Float32
        }
        #[cfg(not(any(
            feature = "accelerate",
            feature = "mkl",
            feature = "mkl-dynamic",
            feature = "ort"
        )))]
        {
            DType::Float16
        }
    }
}
