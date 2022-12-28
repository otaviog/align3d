use align3d::{
    bilateral::{self}
};
use image::GrayImage;
use ndarray::Array2;
use nshare::ToNdarray2;
use num::NumCast;
use rstest::{rstest, fixture};

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

#[fixture]
pub fn bloei_luma16() -> Array2<u16> {
    image::io::Reader::open("tests/data/images/bloei.jpg")
        .unwrap()
        .decode()
        .unwrap()
        .into_luma16()
        .into_ndarray2()
}


#[rstest]
fn test_bilateral_filter(mut bloei_luma16: Array2<u16>) {
    bloei_luma16.iter_mut().for_each(|v| {
        *v /= std::u16::MAX / 255;
    });

    let mut dst_image = bloei_luma16.clone();
    bilateral::bilateral_filter(&bloei_luma16, 100.0, 10.0, &mut dst_image);

    into_image_luma(bloei_luma16)
        .save("tests/outputs/bilateral_filter-input.png")
        .expect("Save ok");

    into_image_luma(dst_image)
        .save("tests/outputs/bilateral_filter-result.png")
        .expect("Save ok");
}

#[rstest]
fn test_bilateral_filter2(mut bloei_luma8: Array2<u8>) {
    let mut dst_image = bloei_luma8.clone();
    bilateral::bilateral_filter(&bloei_luma8, 100.0, 10.0, &mut dst_image);

    into_image_luma(bloei_luma8)
        .save("tests/outputs/bilateral_filter-input.png")
        .expect("Save ok");

    into_image_luma(dst_image)
        .save("tests/outputs/bilateral_filter-result.png")
        .expect("Save ok");
}
