mod fusion;
mod indexmap;
mod surfel_model;

pub use fusion::SurfelFusion;
pub use surfel_model::{SurfelModel, AttrColorMask};

mod surfel_type;
pub use surfel_type::{Surfel, RimageSurfelBuilder};