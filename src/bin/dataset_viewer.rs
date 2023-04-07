use align3d::bin_utils::dataset::create_dataset_from_string;
use align3d::{
    io::dataset::SubsetDataset,
    viz::rgbd_dataset_viewer::RgbdDatasetViewer,
};
use clap::Parser;

#[derive(Parser)]
struct CommandLine {
    // Dataset format: slamtb or ilrgbd
    format: String,
    // Dataset path
    dataset: String,
    // This viewer can only show a subset of the dataset at once.
    // This parameter specifies the number of frames to show.
    // Future versions will come with a better UI.
    #[clap(short, long, default_value = "15")]
    samples: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // TODO: finish this example
    let args = CommandLine::parse();

    let dataset = create_dataset_from_string(args.format, args.dataset).unwrap();
    let dataset = Box::new(SubsetDataset::new(dataset, [0, 15, 30, 45, 60, 75, 90, 120, 160, 250].into()));

    let dataset_viewer = RgbdDatasetViewer::new(dataset);
    dataset_viewer.run();
    Ok(())
}
