use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use super::{Geometry, LoadError};
use ndarray::{Array2, Axis};
use ply_rs::ply::{
    Addable, DefaultElement, ElementDef, Encoding, Ply, Property, PropertyDef, PropertyType,
    ScalarType,
};
use ply_rs::writer::Writer;
use ply_rs::{parser, ply};
// use std::io;

/// We know, what data we want to read, hence we can be more efficient by loading the data into structs.
#[derive(Debug)] // not necessary for parsing, only for println at end of example.
struct Vertex {
    point: [f32; 3],
    normal: [f32; 3],
    color: [u8; 3],
}

#[derive(Debug)]
struct Face {
    vertex_index: Vec<i32>,
}

// The structs need to implement the PropertyAccess trait, otherwise the parser doesn't know how to write to them.
// Most functions have default, hence you only need to implement, what you expect to need.

impl ply::PropertyAccess for Vertex {
    fn new() -> Self {
        Vertex {
            point: [0f32; 3],
            normal: [0f32; 3],
            color: [0u8; 3],
        }
    }
    fn set_property(&mut self, key: String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("x", ply::Property::Float(v)) => self.point[0] = v,
            ("y", ply::Property::Float(v)) => self.point[1] = v,
            ("z", ply::Property::Float(v)) => self.point[2] = v,
            ("nx", ply::Property::Float(v)) => self.normal[0] = v,
            ("ny", ply::Property::Float(v)) => self.normal[1] = v,
            ("nz", ply::Property::Float(v)) => self.normal[2] = v,
            ("red", ply::Property::UChar(v)) => self.color[0] = v,
            ("green", ply::Property::UChar(v)) => self.color[1] = v,
            ("blue", ply::Property::UChar(v)) => self.color[2] = v,
            (_, _) => (), // TODO: Add log
        }
    }
}

// same thing for Face
impl ply::PropertyAccess for Face {
    fn new() -> Self {
        Face {
            vertex_index: Vec::new(),
        }
    }
    fn set_property(&mut self, key: String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("vertex_index" | "vertex_indices", ply::Property::ListInt(vec)) => {
                self.vertex_index = vec
            }
            (k, _) => panic!("Face: Unexpected key/value combination: key: {}", k),
        }
    }
}

pub fn read_ply<P>(filepath: P) -> Result<Geometry, LoadError>
where
    P: AsRef<Path>,
{
    let fptr = std::fs::File::open(filepath)?;
    let mut f = std::io::BufReader::new(fptr);

    let vertex_parser = parser::Parser::<Vertex>::new();
    let header = vertex_parser.read_header(&mut f).unwrap();

    // Depending on the header, read the data into our structs..
    let mut point_array = None;
    let mut normal_array = None;
    let mut color_array = None;
    let mut face_array = None;
    for (_ignore_key, element) in &header.elements {
        match element.name.as_ref() {
            "vertex" => {
                let vertex_parser = parser::Parser::<Vertex>::new();
                let vertex_vec = vertex_parser
                    .read_payload_for_element(&mut f, element, &header)
                    .unwrap();

                point_array = Some(Array2::<f32>::from_shape_fn(
                    (vertex_vec.len(), 3),
                    |(i, c)| vertex_vec[i].point[c],
                ));

                if ["nx", "ny", "nz"]
                    .iter()
                    .all(|k| element.properties.contains_key(*k))
                {
                    normal_array = Some(Array2::<f32>::from_shape_fn(
                        (vertex_vec.len(), 3),
                        |(i, c)| vertex_vec[i].normal[c],
                    ));
                }

                if ["red", "green", "blue"]
                    .iter()
                    .all(|k| element.properties.contains_key(*k))
                {
                    color_array = Some(Array2::<u8>::from_shape_fn(
                        (vertex_vec.len(), 3),
                        |(i, c)| vertex_vec[i].color[c],
                    ));
                }
            }
            "face" => {
                let face_parser = parser::Parser::<Face>::new();
                let face_vec = face_parser
                    .read_payload_for_element(&mut f, element, &header)
                    .unwrap();
                face_array = Some(Array2::<usize>::from_shape_fn(
                    (face_vec.len(), 3),
                    |(i, c)| face_vec[i].vertex_index[c] as usize,
                ));
            }
            _ => panic!("Unexpected element"), // _ => return Err(LoadError::ParseError((format!("Unexpected element {}", _))));
        }
    }

    Ok(Geometry {
        points: point_array.unwrap(),
        colors: color_array,
        normals: normal_array,
        indices: face_array,
        texcoords: None,
    })
}

pub fn write_ply<P>(filepath: P, geom: &Geometry) -> Result<(), std::io::Error>
where
    P: AsRef<Path>,
{
    let mut ply = {
        let mut ply = Ply::<DefaultElement>::new();
        let mut vertex_element = ElementDef::new("vertex".to_string());
        ["x", "y", "z"].iter().for_each(|key| {
            vertex_element.properties.add(PropertyDef::new(
                key.to_string(),
                PropertyType::Scalar(ScalarType::Float),
            ));
        });

        let mut vertex_array: Vec<DefaultElement> = geom
            .points
            .axis_iter(Axis(0))
            .map(|point| {
                let mut elem = DefaultElement::new();
                elem.insert("x".to_string(), Property::Float(point[0]));
                elem.insert("y".to_string(), Property::Float(point[1]));
                elem.insert("z".to_string(), Property::Float(point[2]));
                elem
            })
            .collect();

        if let Some(normals) = &geom.normals {
            ["nx", "ny", "nz"].iter().for_each(|key| {
                vertex_element.properties.add(PropertyDef::new(
                    key.to_string(),
                    PropertyType::Scalar(ScalarType::Float),
                ));
            });

            normals
                .axis_iter(Axis(0))
                .enumerate()
                .for_each(|(i, normal)| {
                    vertex_array[i].insert("nx".to_string(), Property::Float(normal[0]));
                    vertex_array[i].insert("ny".to_string(), Property::Float(normal[1]));
                    vertex_array[i].insert("nz".to_string(), Property::Float(normal[2]));
                });
        }

        if let Some(colors) = &geom.colors {
            ["red", "green", "blue"].iter().for_each(|key| {
                vertex_element.properties.add(PropertyDef::new(
                    key.to_string(),
                    PropertyType::Scalar(ScalarType::UChar),
                ));
            });

            colors
                .axis_iter(Axis(0))
                .enumerate()
                .for_each(|(i, color)| {
                    vertex_array[i].insert("red".to_string(), Property::UChar(color[0]));
                    vertex_array[i].insert("green".to_string(), Property::UChar(color[1]));
                    vertex_array[i].insert("blue".to_string(), Property::UChar(color[2]));
                });
        }

        ply.header.elements.add(vertex_element);
        ply.payload.insert("vertex".to_string(), vertex_array);

        if let Some(indices) = &geom.indices {
            let mut indice_element = ElementDef::new("face".to_string());

            indice_element.properties.add(PropertyDef::new(
                "vertex_indices".to_string(),
                PropertyType::List(ScalarType::UChar, ScalarType::Int),
            ));
            let indice_array: Vec<DefaultElement> = indices
                .axis_iter(Axis(0))
                .map(|face| {
                    let mut elem = DefaultElement::new();
                    elem.insert(
                        "vertex_indices".to_string(),
                        Property::ListInt(face.iter().map(|f| *f as i32).collect()),
                    );
                    elem
                })
                .collect();

            ply.header.elements.add(indice_element);
            ply.payload.insert("face".to_string(), indice_array);
        }

        ply.make_consistent().unwrap();
        ply
    };

    ply.header.encoding = Encoding::Ascii;

    let mut buf = BufWriter::new(File::create(filepath)?);
    Writer::new().write_ply(&mut buf, &mut ply)?;

    Ok(())
}

#[cfg(test)]
mod test {
    use super::{read_ply, write_ply};

    #[test]
    fn should_write_the_same_as_read() {
        let geom = read_ply("tests/data/teapot.ply").unwrap();
        write_ply("tests/data/out-teapot.ply", &geom).unwrap();
    }
}
