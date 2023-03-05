use ndarray::{Array1, Array2, Array3, Axis, s};

pub fn rgb_to_luma(r: u8, g: u8, b: u8) -> f32 {
    const DIV: f32 = 1.0 / 255.0;
    (r as f32 * 0.3 + g as f32 * 0.59 + b as f32 * 0.11) * DIV
}

pub fn rgb_to_luma_u8(r: u8, g: u8, b: u8) -> u8 {
    (r as f32 * 0.3 + g as f32 * 0.59 + b as f32 * 0.11) as u8
}

pub fn convert_rgb_to_luma_u8(image: &Array3<u8>) -> Array2<u8> {
    let (_, h, w) = image.dim();
    //println!("{}", c);
    //let grayscale: Array1<u8> = image
    //    .view()
    //    .into_shape((3, h * w))
    //    .unwrap()
    //    .axis_iter(Axis(1))
    //    .map(|pixel| rgb_to_luma_u8(pixel[0], pixel[1], pixel[2]))
    //    .collect();
//
    //grayscale.into_shape((h, w)).unwrap()

    let grayscale: Array1<u8> = image
        .slice(s![0, .., ..]).iter()
        .zip(image.slice(s![1, .., ..]).iter())
        .zip(image.slice(s![2, .., ..]).iter())
        .map(|((r, g), b)| {
            rgb_to_luma_u8(*r, *g, *b)
        }).collect();
    
    grayscale.into_shape((h, w)).unwrap()
}
