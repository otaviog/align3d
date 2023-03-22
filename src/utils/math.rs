pub fn integer_fractional(x: f32) -> (usize, f32) {
    let x_int = x as usize;
    let x_frac = x.fract();

    (x_int, x_frac)
}