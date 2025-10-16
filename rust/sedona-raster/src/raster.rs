use arrow::array::{ArrayRef, BinaryBuilder, ListBuilder, StructArray, StructBuilder};
use arrow::buffer::MutableBuffer;
use arrow::datatypes::{DataType, Field, FieldRef, Fields};
use arrow::error::ArrowError;
use std::sync::Arc;

#[repr(u16)]
#[derive(Clone, Debug)]
pub enum BandDataType {
    UInt8 = 0,
    UInt16 = 1,
    Int16 = 2,
    UInt32 = 3,
    Int32 = 4,
    Float32 = 5,
    Float64 = 6,
    // Consider support for complex types for scientific data
}

/// Storage strategy for raster band data within Apache Arrow arrays.
/// 
/// This enum defines how raster data is physically stored and accessed:
/// 
/// **InDb**: Raster data is embedded directly in the Arrow array as binary blobs.
///   - Pros: Self-contained, no external dependencies, fast access for small-medium rasters
///   - Cons: Increases Arrow array size, memory usage grows with raster size
///   - Best for: Tiles, thumbnails, processed results, small rasters (<10MB per band)
/// 
/// **OutDbRef**: Raster data is stored externally with references in the Arrow array.
///   - Pros: Keeps Arrow arrays lightweight, supports massive rasters, enables lazy loading
///   - Cons: Requires external storage management, potential for broken references
///   - Best for: Large satellite imagery, time series data, cloud-native workflows
///   - Reference format: JSON with storage type, path/URL, credentials, metadata
///   - Supported backends: S3, GCS, Azure Blob, local filesystem, HTTP endpoints
#[repr(u16)]
#[derive(Clone, Debug)]
pub enum StorageType {
    InDb = 0,
    OutDbRef = 1,
}

pub struct RasterSchema;

impl RasterSchema {
    // Raster schema:
    pub fn fields() -> Fields {
        Fields::from(vec![
            Field::new(column::METADATA, Self::metadata_type(), false),
            Field::new(column::BANDS, Self::bands_type(), true),
        ])
    }

    /// Raster metadata schema
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

    /// Bounding box schema
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

    /// Individual band schema
    pub fn band_type() -> DataType {
        DataType::Struct(Fields::from(vec![
            Field::new(column::METADATA, Self::band_metadata_type(), false),
            Field::new(column::DATA, Self::band_data_type(), false),
        ]))
    }

    /// Band metadata schema
    pub fn band_metadata_type() -> DataType {
        DataType::Struct(Fields::from(vec![
            Field::new(column::NODATAVALUE, DataType::Binary, false),
            Field::new(column::STORAGE_TYPE, DataType::UInt32, false),
            Field::new(column::DATATYPE, DataType::UInt32, false),
        ]))
    }

    /// Band data schema (single binary blob)
    pub fn band_data_type() -> DataType {
        DataType::Binary
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

/// Builder for constructing raster arrays with zero-copy band data writing
pub struct RasterBuilder {
    metadata_builder: StructBuilder,
    bands_builder: ListBuilder<StructBuilder>,
}

impl RasterBuilder {
    /// Create a new raster builder with the specified capacity
    pub fn new(capacity: usize) -> Self {
        let metadata_builder = StructBuilder::from_fields(
            match RasterSchema::metadata_type() {
                DataType::Struct(fields) => fields,
                _ => panic!("Expected struct type for metadata"),
            },
            capacity,
        );

        let bands_builder = ListBuilder::new(StructBuilder::from_fields(
            match RasterSchema::band_type() {
                DataType::Struct(fields) => fields,
                _ => panic!("Expected struct type for band"),
            },
            0, // Initial capacity for bands
        ));

        Self {
            metadata_builder,
            bands_builder,
        }
    }

    /// Start a new raster and write its metadata
    pub fn start_raster(&mut self, metadata: RasterMetadata) -> Result<(), ArrowError> {
        self.append_metadata(metadata)
    }

    /// Get direct access to the BinaryBuilder for writing the current band's data
    pub fn band_data_writer(&mut self) -> &mut BinaryBuilder {
        let band_builder = self.bands_builder.values();
        band_builder.field_builder::<BinaryBuilder>(1).unwrap()
    }

    /// Create a MutableBuffer that can be written to directly
    pub fn create_band_buffer(
        &mut self,
        capacity: usize,
    ) -> (MutableBuffer, impl FnOnce(MutableBuffer) + '_) {
        let mut buffer = MutableBuffer::with_capacity(capacity);

        // Pre-allocate the buffer to the exact size
        buffer.resize(capacity, 0);

        let commit = move |buffer: MutableBuffer| {
            // Convert MutableBuffer to &[u8] and append to BinaryBuilder
            let data = buffer.as_slice();
            self.band_data_writer().append_value(data);
        };

        (buffer, commit)
    }

    /// Alternative: Get a mutable slice from a MutableBuffer for GDAL
    /// This provides the most direct access for zero-copy operations
    /// TODO: have this 3 different way.... pick one!!
    pub fn get_band_buffer_slice(&mut self, size: usize) -> (MutableBuffer, &mut [u8]) {
        let mut buffer = MutableBuffer::with_capacity(size);
        buffer.resize(size, 0);

        // Get mutable slice that GDAL can write to
        let slice = unsafe {
            // This is safe because we just allocated the buffer with the exact size
            std::slice::from_raw_parts_mut(buffer.as_mut_ptr(), size)
        };

        (buffer, slice)
    }

    /// Commit a MutableBuffer to the band data
    pub fn commit_band_buffer(&mut self, buffer: MutableBuffer) {
        let data = buffer.as_slice();
        self.band_data_writer().append_value(data);
    }

    /// Finish writing the current band with its metadata
    /// TODO: The band_metadata is in the finish in the band call, but in the
    /// start in the raster call. Make it consistent.
    pub fn finish_band(&mut self, band_metadata: BandMetadata) -> Result<(), ArrowError> {
        let band_builder = self.bands_builder.values();

        let metadata_builder = band_builder.field_builder::<StructBuilder>(0).unwrap();

        if let Some(nodata) = band_metadata.nodata_value {
            metadata_builder
                .field_builder::<BinaryBuilder>(0)
                .unwrap()
                .append_value(&nodata);
        } else {
            metadata_builder
                .field_builder::<BinaryBuilder>(0)
                .unwrap()
                .append_null();
        }

        metadata_builder
            .field_builder::<arrow::array::UInt32Builder>(1)
            .unwrap()
            .append_value(band_metadata.storage_type as u32);

        metadata_builder
            .field_builder::<arrow::array::UInt32Builder>(2)
            .unwrap()
            .append_value(band_metadata.datatype as u32);

        metadata_builder.append(true);

        // Finish the band
        band_builder.append(true);
        Ok(())
    }

    /// Finish all bands for the current raster
    pub fn finish_raster(&mut self) -> Result<(), ArrowError> {
        self.bands_builder.append(true);
        Ok(())
    }

    /// Append raster metadata
    fn append_metadata(&mut self, metadata: RasterMetadata) -> Result<(), ArrowError> {
        // Width
        self.metadata_builder
            .field_builder::<arrow::array::UInt64Builder>(0)
            .unwrap()
            .append_value(metadata.width);

        // Height
        self.metadata_builder
            .field_builder::<arrow::array::UInt64Builder>(1)
            .unwrap()
            .append_value(metadata.height);

        // Geotransform parameters
        self.metadata_builder
            .field_builder::<arrow::array::Float64Builder>(2)
            .unwrap()
            .append_value(metadata.upperleft_x);

        self.metadata_builder
            .field_builder::<arrow::array::Float64Builder>(3)
            .unwrap()
            .append_value(metadata.upperleft_y);

        self.metadata_builder
            .field_builder::<arrow::array::Float64Builder>(4)
            .unwrap()
            .append_value(metadata.scale_x);

        self.metadata_builder
            .field_builder::<arrow::array::Float64Builder>(5)
            .unwrap()
            .append_value(metadata.scale_y);

        self.metadata_builder
            .field_builder::<arrow::array::Float64Builder>(6)
            .unwrap()
            .append_value(metadata.skew_x);

        self.metadata_builder
            .field_builder::<arrow::array::Float64Builder>(7)
            .unwrap()
            .append_value(metadata.skew_y);

        // Optional bounding box
        if let Some(bbox) = metadata.bounding_box {
            let bbox_builder = self
                .metadata_builder
                .field_builder::<StructBuilder>(8)
                .unwrap();

            bbox_builder
                .field_builder::<arrow::array::Float64Builder>(0)
                .unwrap()
                .append_value(bbox.min_x);

            bbox_builder
                .field_builder::<arrow::array::Float64Builder>(1)
                .unwrap()
                .append_value(bbox.min_y);

            bbox_builder
                .field_builder::<arrow::array::Float64Builder>(2)
                .unwrap()
                .append_value(bbox.max_x);

            bbox_builder
                .field_builder::<arrow::array::Float64Builder>(3)
                .unwrap()
                .append_value(bbox.max_y);

            bbox_builder.append(true);
        }

        self.metadata_builder.append(true);

        Ok(())
    }

    /// Append a null raster
    pub fn append_null(&mut self) -> Result<(), ArrowError> {
        self.metadata_builder.append(false);
        self.bands_builder.append(false);
        Ok(())
    }

    /// Finish building and return the constructed StructArray
    pub fn finish(mut self) -> Result<StructArray, ArrowError> {
        let metadata_array = self.metadata_builder.finish();
        let bands_array = self.bands_builder.finish();

        let fields = RasterSchema::fields();
        let arrays: Vec<ArrayRef> = vec![Arc::new(metadata_array), Arc::new(bands_array)];

        Ok(StructArray::new(fields, arrays, None))
    }
}

/// Convenience wrapper for the zero-copy band writing approach
impl RasterBuilder {
    /// High-level method that allows for zero-copy with a callback approach
    pub fn append_raster_with_callback<F>(
        &mut self,
        metadata: RasterMetadata,
        band_count: usize,
        mut write_bands: F,
    ) -> Result<(), ArrowError>
    where
        F: FnMut(usize, &mut BinaryBuilder) -> Result<BandMetadata, ArrowError>,
    {
        self.start_raster(metadata)?;

        for band_index in 0..band_count {
            let band_metadata = {
                let binary_builder = self.band_data_writer();
                write_bands(band_index, binary_builder)?
            };
            self.finish_band(band_metadata)?;
        }

        self.finish_raster()?;
        Ok(())
    }
}

/// Metadata for a raster
#[derive(Debug, Clone)]
pub struct RasterMetadata {
    pub width: u64,
    pub height: u64,
    pub upperleft_x: f64,
    pub upperleft_y: f64,
    pub scale_x: f64,
    pub scale_y: f64,
    pub skew_x: f64,
    pub skew_y: f64,
    pub bounding_box: Option<BoundingBox>,
}

/// Bounding box coordinates
#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

/// Metadata for a single band
#[derive(Debug, Clone)]
pub struct BandMetadata {
    pub nodata_value: Option<Vec<u8>>,
    pub storage_type: StorageType,
    pub datatype: BandDataType,
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;
    use arrow::datatypes::DataType;

    #[test]
    fn test_raster_builder_basic() {
        let mut builder = RasterBuilder::new(1);

        let metadata = RasterMetadata {
            width: 100,
            height: 100,
            upperleft_x: -120.0,
            upperleft_y: 40.0,
            scale_x: 0.1,
            scale_y: -0.1,
            skew_x: 0.0,
            skew_y: 0.0,
            bounding_box: Some(BoundingBox {
                min_x: -120.0,
                min_y: 30.0,
                max_x: -110.0,
                max_y: 40.0,
            }),
        };

        // Start writing a raster
        builder.start_raster(metadata).unwrap();

        // Write band with direct BinaryBuilder access
        {
            let raster_data = vec![1u8; 10000]; // 100x100 raster
            builder.band_data_writer().append_value(&raster_data);

            // Finish the band with metadata
            builder
                .finish_band(BandMetadata {
                    nodata_value: Some(vec![255]),
                    storage_type: StorageType::InDb,
                    datatype: BandDataType::UInt8,
                })
                .unwrap();
        }

        builder.finish_raster().unwrap();

        let result = builder.finish().unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result.num_columns(), 2);
    }

    #[test]
    fn test_mutable_buffer_gdal_integration() {
        let mut builder = RasterBuilder::new(1);

        let metadata = RasterMetadata {
            width: 256,
            height: 256,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            skew_x: 0.0,
            skew_y: 0.0,
            bounding_box: None,
        };

        builder.start_raster(metadata).unwrap();

        // GDAL integration pattern with MutableBuffer
        {
            let buffer_size = 256 * 256; // width * height for UInt8 data
            let (mut buffer, commit) = builder.create_band_buffer(buffer_size);

            // Simulate GDAL reading directly into the MutableBuffer
            // In real code: gdal_dataset.read_into_buffer(buffer.as_mut_slice())?
            simulate_gdal_read_into_buffer(&mut buffer);

            // Commit the buffer to Arrow
            commit(buffer);
        }

        builder
            .finish_band(BandMetadata {
                nodata_value: None,
                storage_type: StorageType::InDb,
                datatype: BandDataType::UInt8,
            })
            .unwrap();

        builder.finish_raster().unwrap();
        let result = builder.finish().unwrap();

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_mutable_buffer_slice_pattern() {
        let mut builder = RasterBuilder::new(1);

        let metadata = RasterMetadata {
            width: 100,
            height: 100,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            skew_x: 0.0,
            skew_y: 0.0,
            bounding_box: None,
        };

        builder.start_raster(metadata).unwrap();

        // Alternative pattern: get slice directly
        {
            let buffer_size = 10000; // 100x100
            let (buffer, slice) = builder.get_band_buffer_slice(buffer_size);

            // GDAL can write directly to this slice
            // gdal_dataset.read_into_slice(slice)?
            for (i, byte) in slice.iter_mut().enumerate() {
                *byte = (i % 256) as u8;
            }

            // Commit the buffer
            builder.commit_band_buffer(buffer);
        }

        builder
            .finish_band(BandMetadata {
                nodata_value: Some(vec![255]),
                storage_type: StorageType::InDb,
                datatype: BandDataType::UInt8,
            })
            .unwrap();

        builder.finish_raster().unwrap();
        let result = builder.finish().unwrap();

        assert_eq!(result.len(), 1);
    }

    // Helper function to simulate GDAL reading into a MutableBuffer
    fn simulate_gdal_read_into_buffer(buffer: &mut MutableBuffer) {
        let slice = unsafe { std::slice::from_raw_parts_mut(buffer.as_mut_ptr(), buffer.len()) };
        for (i, byte) in slice.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
    }

    #[test]
    fn test_raster_builder_callback_approach() {
        let mut builder = RasterBuilder::new(1);

        let metadata = RasterMetadata {
            width: 50,
            height: 50,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            skew_x: 0.0,
            skew_y: 0.0,
            bounding_box: None,
        };

        // Use callback approach for cleaner API
        builder
            .append_raster_with_callback(metadata, 2, |band_index, binary_builder| {
                match band_index {
                    0 => {
                        // Write RGB band - direct access to BinaryBuilder
                        let rgb_data = vec![255u8; 2500]; // 50x50 RGB values
                        binary_builder.append_value(&rgb_data);
                        Ok(BandMetadata {
                            nodata_value: None,
                            storage_type: StorageType::InDb,
                            datatype: BandDataType::UInt8,
                        })
                    }
                    1 => {
                        // Write NIR band - direct access to BinaryBuilder
                        let nir_data = vec![128u8; 2500]; // 50x50 NIR values
                        binary_builder.append_value(&nir_data);
                        Ok(BandMetadata {
                            nodata_value: Some(vec![0]),
                            storage_type: StorageType::InDb,
                            datatype: BandDataType::UInt8,
                        })
                    }
                    _ => unreachable!(),
                }
            })
            .unwrap();

        let result = builder.finish().unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_large_raster_simulation() {
        let mut builder = RasterBuilder::new(1);

        let metadata = RasterMetadata {
            width: 10000, // Large raster
            height: 10000,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            skew_x: 0.0,
            skew_y: 0.0,
            bounding_box: None,
        };

        builder.start_raster(metadata).unwrap();

        // Simulate writing a huge raster band directly
        {
            let mut band_writer = builder.start_band();

            // Direct access to BinaryBuilder - can write massive amounts of data
            let binary_builder = band_writer.data_writer();

            // Reserve space for large raster (100MB)
            let raster_size = 10000 * 10000;
            binary_builder.reserve_exact(raster_size);

            // In real usage, this could be streaming from GDAL or reading from disk
            // Write the entire raster in one operation
            let large_raster_data = vec![42u8; raster_size];
            binary_builder.append_value(&large_raster_data);

            band_writer
                .finish_band(BandMetadata {
                    nodata_value: None,
                    storage_type: StorageType::OutDbRef,
                    datatype: BandDataType::UInt8,
                })
                .unwrap();
        }

        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let result = builder.finish().unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_raster_metadata_struct() {
        let metadata = RasterMetadata {
            width: 256,
            height: 256,
            upperleft_x: -180.0,
            upperleft_y: 90.0,
            scale_x: 0.5,
            scale_y: -0.5,
            skew_x: 0.1,
            skew_y: 0.1,
            bounding_box: Some(BoundingBox {
                min_x: -180.0,
                min_y: -90.0,
                max_x: 180.0,
                max_y: 90.0,
            }),
        };

        assert_eq!(metadata.width, 256);
        assert_eq!(metadata.height, 256);
        assert!(metadata.bounding_box.is_some());
    }

    #[test]
    fn test_band_metadata_struct() {
        let band_metadata = BandMetadata {
            nodata_value: Some(vec![0, 0]),
            storage_type: StorageType::InDb,
            datatype: BandDataType::UInt16,
        };

        assert!(band_metadata.nodata_value.is_some());
        assert_eq!(band_metadata.storage_type as u16, 0);
        assert_eq!(band_metadata.datatype as u16, 1);
    }

    #[test]
    fn test_multiple_bands_zero_copy() {
        let mut builder = RasterBuilder::new(1);

        let metadata = RasterMetadata {
            width: 10,
            height: 10,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            skew_x: 0.0,
            skew_y: 0.0,
            bounding_box: None,
        };

        builder.start_raster(metadata).unwrap();

        // First band
        {
            let mut band_writer = builder.start_band();
            band_writer.data_writer().append_value(&[1, 2, 3]);
            band_writer
                .finish_band(BandMetadata {
                    nodata_value: None,
                    storage_type: StorageType::InDb,
                    datatype: BandDataType::UInt8,
                })
                .unwrap();
        }
        builder.finish_band().unwrap();

        // Second band
        {
            let mut band_writer = builder.start_band();
            band_writer.data_writer().append_value(&[4, 5, 6]);
            band_writer
                .finish_band(BandMetadata {
                    nodata_value: Some(vec![255]),
                    storage_type: StorageType::InDb,
                    datatype: BandDataType::UInt8,
                })
                .unwrap();
        }
        builder.finish_band().unwrap();

        builder.finish_raster().unwrap();

        let result = builder.finish().unwrap();
        assert_eq!(result.len(), 1);
    }

    // Existing schema tests...
    #[test]
    fn test_raster_schema_fields() {
        let fields = RasterSchema::fields();
        assert_eq!(fields.len(), 2);

        // Check metadata field
        let metadata_field = &fields[0];
        assert_eq!(metadata_field.name(), "metadata");
        assert!(!metadata_field.is_nullable());

        // Check bands field
        let bands_field = &fields[1];
        assert_eq!(bands_field.name(), "bands");
        assert!(bands_field.is_nullable());
    }
}
