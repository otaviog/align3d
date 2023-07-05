mod fusion;
mod indexmap;
mod surfel_model;

pub use fusion::{SurfelFusion, SurfelFusionParameters, RangeImage2};
pub use surfel_model::{AttrColorMaskAge, AttrNormalRadius, AttrPositionConfidence, SurfelModel};

mod surfel_type;
pub use surfel_type::{RimageSurfelBuilder, Surfel};
