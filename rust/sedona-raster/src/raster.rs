use crate::raster::column::{DATA, METADATA};
use arrow::array::{
    Array, ArrayAccessor, ArrayBuilder, ArrayData, AsArray, BinaryBuilder, ListBuilder,
    StructArray, StructBuilder, UInt32Builder,
};
use arrow::array::{BinaryArray, ListArray, UInt32Array};
use arrow::datatypes::{DataType, Field, FieldRef, Fields, ToByteSlice};
use arrow::error::ArrowError;

#[repr(u16)]
pub enum BandDataType {
    UInt8 = 0,
    UInt16 = 1,
    Int16 = 2,
    UInt32 = 3,
    Int32 = 4,
    Float32 = 5,
    Float64 = 6,
    // Add complex types if needed
}

#[repr(u16)]
pub enum StorageType {
    InDb = 0,
    OutDbRef = 1,
}

/// Raster schema definition utilities
pub struct RasterSchema;

impl RasterSchema {
    // Raster schema:
    pub fn fields() -> Fields {
        Fields::from(vec![
            Field::new(column::METADATA, Self::metadata_type(), false),
            Field::new(column::BANDS, Self::bands_type(), true),
        ])
    }

    /// Raster metadata schema (dimensions and geospatial transformation)
    pub fn metadata_type() -> DataType {
        DataType::Struct(Fields::from(vec![
            // Raster dimensions
            Field::new(column::WIDTH, DataType::UInt64, false),
            Field::new(column::HEIGHT, DataType::UInt64, false),
            // Geospatial transformation parameters
            Field::new(column::UPPERLEFT_X, DataType::Float64, false),
            Field::new(column::UPPERLEFT_Y, DataType::Float64, false),
            Field::new(column::SCALE_X, DataType::Float64, false),
            Field::new(column::SCALE_Y, DataType::Float64, false),
            Field::new(column::SKEW_X, DataType::Float64, false),
            Field::new(column::SKEW_Y, DataType::Float64, false),
            // Optional bounding box
            Field::new(column::BOUNDING_BOX, Self::bounding_box_type(), true),
        ]))
    }

    /// Bounding box schema (min_x, min_y, max_x, max_y)
    pub fn bounding_box_type() -> DataType {
        DataType::Struct(Fields::from(vec![
            Field::new(column::MIN_X, DataType::Float64, false),
            Field::new(column::MIN_Y, DataType::Float64, false),
            Field::new(column::MAX_X, DataType::Float64, false),
            Field::new(column::MAX_Y, DataType::Float64, false),
        ]))
    }

    /// Bands list schema
    pub fn bands_type() -> DataType {
        DataType::List(FieldRef::new(Field::new(
            column::BAND,
            Self::band_type(),
            false,
        )))
    }

    /// Individual band schema (metadata + data)
    pub fn band_type() -> DataType {
        DataType::Struct(Fields::from(vec![
            Field::new(column::METADATA, Self::band_metadata_type(), false),
            Field::new(column::DATA, Self::band_data_type(), false),
        ]))
    }

    /// Band metadata schema (nodata, storage type, data type)
    /// Con
    pub fn band_metadata_type() -> DataType {
        DataType::Struct(Fields::from(vec![
            Field::new(column::NODATAVALUE, DataType::Binary, false),
            Field::new(column::STORAGE_TYPE, DataType::UInt32, false),
            Field::new(column::DATATYPE, DataType::UInt32, false),
        ]))
    }

    /// Band data schema (list of binary chunks)
    pub fn band_data_type() -> DataType {
        DataType::List(FieldRef::new(Field::new(
            column::DATA,
            DataType::Binary,
            false,
        )))
    }
}

pub mod column {
    pub const METADATA: &str = "metadata";
    pub const BANDS: &str = "bands";
    pub const BAND: &str = "band";
    pub const DATA: &str = "data";
    
    // Raster metadata fields
    pub const WIDTH: &str = "width";
    pub const HEIGHT: &str = "height";
    pub const UPPERLEFT_X: &str = "upperleft_x";
    pub const UPPERLEFT_Y: &str = "upperleft_y";
    pub const SCALE_X: &str = "scale_x";
    pub const SCALE_Y: &str = "scale_y";
    pub const SKEW_X: &str = "skew_x";
    pub const SKEW_Y: &str = "skew_y";
    pub const BOUNDING_BOX: &str = "bounding_box";
    
    // Bounding box fields
    pub const MIN_X: &str = "min_x";
    pub const MIN_Y: &str = "min_y";
    pub const MAX_X: &str = "max_x";
    pub const MAX_Y: &str = "max_y";
    
    // Band metadata fields
    pub const NODATAVALUE: &str = "nodata_value";
    pub const STORAGE_TYPE: &str = "storage_type";
    pub const DATATYPE: &str = "data_type";
}