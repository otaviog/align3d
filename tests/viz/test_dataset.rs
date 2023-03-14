use align3d::{
    io::{core::{SubsetDataset, RGBDDataset}, slamtb::SlamTbDataset},
    viz::rgbd_dataset_viewer::RgbdDatasetViewer,
};

pub fn main() {
    let dataset = SlamTbDataset::load("tests/data/rgbd/sample1").unwrap();
    println!("{:?} images loaded", dataset.trajectory().unwrap());
    
    let len_ds = dataset.len();
    let dataset = SubsetDataset::new(
        dataset,
        [0, len_ds / 2,  len_ds - 1].into(),
    );
    println!("{:?} images loaded", dataset.trajectory().unwrap());
    RgbdDatasetViewer::new(Box::new(dataset)).run();
}
