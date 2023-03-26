use align3d::{
    io::dataset::{IndoorLidarDataset, RgbdDataset, SlamTbDataset, SubsetDataset},
    viz::rgbd_dataset_viewer::RgbdDatasetViewer,
};
use clap::Parser;

#[derive(Debug, PartialEq, Clone, Copy)]
enum DatasetFormat {
    SlamTb,
    IlRgbd,
}

fn validate_format(format: String) -> Result<DatasetFormat, String> {
    match format.as_str() {
        "slamtb" => Ok(DatasetFormat::SlamTb),
        "ilrgbd" => Ok(DatasetFormat::IlRgbd),
        _ => Err(String::from(format!("Invalid dataset format: {format}"))),
    }
}

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
    let args = CommandLine::parse();

    let format = validate_format(args.format)?;
    let dataset: Box<dyn RgbdDataset> = match format {
        DatasetFormat::SlamTb => Box::new(SlamTbDataset::load(&args.dataset)?),
        DatasetFormat::IlRgbd => Box::new(IndoorLidarDataset::load(&args.dataset)?),
    };

    //let len_ds = dataset.len();

    let dataset = Box::new(SubsetDataset::new(
        dataset,
        [0, 15, 30].into(),
    ));

    let dataset_viewer = RgbdDatasetViewer::new(dataset);
    dataset_viewer.run();
    Ok(())
}
