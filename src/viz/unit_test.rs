use rstest::fixture;

use crate::viz::Manager;

#[fixture]
pub fn vk_manager() -> Manager {
    Manager::default()
}
