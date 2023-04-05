use align3d::{
    io::dataset::{RgbdDataset, SlamTbDataset, SubsetDataset},
    viz::rgbd_dataset_viewer::RgbdDatasetViewer,
};

pub fn main() {
    let dataset = SlamTbDataset::load("tests/data/rgbd/sample2").unwrap();
    let dataset = SlamTbDataset::load("tests/data/rgbd/sample2").unwrap();
    let len_ds = dataset.len();
    let dataset = SubsetDataset::new(Box::new(dataset), [0, len_ds / 2, len_ds - 1].into());
    RgbdDatasetViewer::new(Box::new(dataset)).run();
}
