use align3d::{bilateral::BilateralFilter, Array2Recycle};
use image::GrayImage;
use ndarray::Array2;
use nshare::ToNdarray2;
use num::NumCast;
use rstest::{fixture, rstest};

fn into_image_luma<Luma>(image: Array2<Luma>) -> GrayImage
where
    Luma: NumCast,
{
    GrayImage::from_vec(
        image.ncols() as u32,
        image.nrows() as u32,
        image
            .into_raw_vec()
            .iter()
            .map(|v| v.to_u8().unwrap())
            .collect::<Vec<u8>>(),
    )
    .unwrap()
}

#[fixture]
pub fn bloei_luma8() -> Array2<u8> {
    image::io::Reader::open("tests/data/images/bloei.jpg")
        .unwrap()
        .decode()
        .unwrap()
        .into_luma8()
        .into_ndarray2()
}

#[rstest]
fn test_bilateral_filter(bloei_luma8: Array2<u8>) {
    let filtered_image =
        BilateralFilter::new(100.0, 10.0).filter(&bloei_luma8, Array2Recycle::Empty);

    into_image_luma(bloei_luma8)
        .save("tests/outputs/bilateral_filter-input.png")
        .expect("Save ok");

    into_image_luma(filtered_image.clone())
        .save("tests/outputs/bilateral_filter-result.png")
        .expect("Save ok");
}
