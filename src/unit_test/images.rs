use ndarray::{Array2, Array3};
use nshare::ToNdarray2;
use rstest::fixture;

use crate::image::IntoArray3;

#[fixture]
pub fn bloei_rgb() -> Array3<u8> {
    image::io::Reader::open("tests/data/images/bloei.jpg")
        .unwrap()
        .decode()
        .unwrap()
        .into_rgb8()
        .into_array3()
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

#[fixture]
pub fn bloei_luma16() -> Array2<u16> {
    let mut image = image::io::Reader::open("tests/data/images/bloei.jpg")
        .unwrap()
        .decode()
        .unwrap()
        .into_luma16()
        .into_ndarray2();

    image.iter_mut().for_each(|v| {
        *v /= std::u16::MAX / 5000;
    });
    image
}
