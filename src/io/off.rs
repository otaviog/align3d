use ndarray::prelude::*;
use num::Zero;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::num::ParseIntError;
use std::str::FromStr;

use super::{Geometry, LoadError};

struct TextParserContext {
    buf_reader: BufReader<File>,
    filepath: String,
    line_count: usize,
}

impl TextParserContext {
    /// Reads a line and increase the line counter. It already trim the string.
    fn read_line(&mut self) -> Result<String, LoadError> {
        let mut line = String::new();
        self.buf_reader.read_line(&mut line)?;
        self.line_count += 1;
        return Ok(line.trim().to_string());
    }

    /// Formats an error message by putting the file name, the current line and the supplied message.
    ///
    /// # Arguments
    ///
    /// * `message` - An error message.
    fn gen_error(&self, message: String) -> LoadError {
        LoadError::ParseError(format!(
            "{}:{}: {}",
            self.filepath, self.line_count, message
        ))
    }
}

fn read_off_elements<T: Clone + FromStr + Zero + Copy>(
    num_elements: usize,
    parser_context: &mut TextParserContext,
) -> Result<Array2<T>, LoadError> {
    let mut elements = Array2::<T>::zeros((num_elements, 3));
    for i in 0..num_elements {
        let line = parser_context.read_line()?;
        if let [Ok(x), Ok(y), Ok(z)] =
            line.split(' ').map(|x| x.parse::<T>()).collect::<Vec<_>>()[..]
        {
            elements[[i, 0]] = x;
            elements[[i, 1]] = y;
            elements[[i, 2]] = z;
        } else {
            return Err(parser_context.gen_error(format!("Invalid vertex. Got `{line}`")));
        }
    }

    Ok(elements)
}

fn read_off_faces(
    num_elements: usize,
    parser_context: &mut TextParserContext,
) -> Result<Array2<usize>, LoadError> {
    let mut faces = Vec::<usize>::new();
    faces.reserve(num_elements * 4);

    for _ in 0..num_elements {
        let line = parser_context.read_line()?;

        let indices = line
            .split(' ')
            .map(|x| x.parse::<usize>())
            .collect::<Vec<_>>();

        match indices[..] {
            [Ok(_), Ok(f1), Ok(f2), Ok(f3)] => {
                faces.push(f1);
                faces.push(f2);
                faces.push(f3);
            }
            [Ok(_), Ok(f1), Ok(f2), Ok(f3), Ok(f4)] => {
                faces.push(f1);
                faces.push(f2);
                faces.push(f3);

                faces.push(f4);
                faces.push(f1);
                faces.push(f3);
            }
            _ => {
                return Err(parser_context.gen_error(format!("Invalid face. Got `{line}`")));
            }
        }
    }

    Ok(Array2::from_shape_vec((faces.len() / 3, 3), faces).unwrap())
}

pub fn read_off(filepath: &str) -> Result<Geometry, LoadError> {
    let file = File::open(filepath)?; // .expect("File not found");

    let mut parser_context = TextParserContext {
        buf_reader: std::io::BufReader::new(file),
        filepath: filepath.to_string(),
        line_count: 1,
    };

    let header = parser_context.read_line()?;
    if header.trim() != "OFF" {
        return Err(parser_context.gen_error(format!(
            "file header does not start with 'OFF', got '{header}' instead"
        )));
    }

    let dims = parser_context.read_line()?;
    let values = dims
        .split(' ')
        .map(|x| x.parse::<usize>())
        .collect::<Vec<Result<usize, ParseIntError>>>();

    let (num_verts, num_faces, _) = if let [Ok(v0), Ok(v1), Ok(v2)] = values[..] {
        (v0, v1, v2)
    } else {
        return Err(parser_context.gen_error(format!("Invalid size formats. Got `{dims}`")));
    };

    let vertices = read_off_elements::<f32>(num_verts, &mut parser_context)?;
    let faces = read_off_faces(num_faces, &mut parser_context)?;

    Ok(Geometry {
        points: vertices,
        colors: None,
        normals: None,
        faces: Some(faces),
        texcoords: None,
    })
}

#[cfg(test)]
mod tests {
    use ndarray::Axis;

    #[test]
    fn test_read_off() {
        use super::read_off;
        let geom = read_off("tests/data/teapot.off").expect("Unable to read .off file");
        assert_eq!(geom.points.len_of(Axis(0)), 400);
    }
}
