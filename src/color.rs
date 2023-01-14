pub fn rgb_to_luma(r: u8, g: u8, b: u8) -> f32 {
    const DIV: f32 = 1.0 / 255.0;
    (r as f32 * 0.3 + g as f32 * 0.59 + b as f32 * 0.11) * DIV
}
