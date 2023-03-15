use rerun::Session;

mod data;
use data::sample_rgbd_pointcloud;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pcl = sample_rgbd_pointcloud();
    let mut session = Session::new();
    pcl.rerun_msg("PCL")?.send(&mut session)?;

    session.show()?;

    Ok(())
}
