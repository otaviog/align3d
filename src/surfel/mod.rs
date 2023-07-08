mod fusion;
mod indexmap;
mod surfel_model;

pub use fusion::{SurfelFusion, SurfelFusionParameters};
pub use surfel_model::{
    SurfelModel, VkSurfelColorMaskAge, VkSurfelNormalRadius, VkSurfelPositionConf,
};

mod surfel_type;
pub use surfel_type::{Surfel, SurfelBuilder};
