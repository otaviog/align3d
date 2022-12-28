use align3d::bilateral;
use image::{ImageBuffer, Luma};
use ndarray::Array2;
use nshare::{ToImageLuma, ToNdarray2};

#[test]
fn test_bilateral_filter() {
    let img = image::io::Reader::open("tests/data/images/bloei.jpg")
        .unwrap()
        .decode()
        .unwrap()
        .into_luma16();
    let mut img: Array2<u16> = img.into_ndarray2();
    img.iter_mut().for_each(|v| {
        *v /= std::u16::MAX/255;
    });
    let mut dst = img.clone();
    bilateral::bilateral_filter(&img, 100.0, 10.0, &mut dst);

    let dst_img = image::GrayImage::from_vec(
        dst.ncols() as u32,
        dst.nrows() as u32,
        dst.into_raw_vec()
            .iter()
            .map(|v| *v as u8)
            .collect::<Vec<u8>>(),
    )
    .unwrap();

    dst_img.save("a.png").expect("Save Ok");

    let img = image::GrayImage::from_vec(
        img.ncols() as u32,
        img.nrows() as u32,
        img.into_raw_vec()
            .iter()
            .map(|v| *v as u8)
            .collect::<Vec<u8>>(),
    )
    .unwrap();
    img.save("b.png").expect("Save ok");

}
