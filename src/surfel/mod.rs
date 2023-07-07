mod fusion;
mod indexmap;
mod surfel_model;

pub use fusion::{SurfelFusion, SurfelFusionParameters, RangeImage2};
pub use surfel_model::{VkSurfelColorMaskAge, VkSurfelNormalRadius, VkSurfelPositionConf, SurfelModel};

mod surfel_type;
pub use surfel_type::{SurfelBuilder, Surfel};
