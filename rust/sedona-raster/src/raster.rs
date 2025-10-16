use arrow::array::{
    Array, ArrayRef, BinaryArray, BinaryBuilder, StringViewArray, ListArray, ListBuilder,
    StructArray, StructBuilder, UInt32Array, UInt64Array,
};
use arrow::buffer::MutableBuffer;
use arrow::datatypes::{DataType, Field, FieldRef, Fields};
use arrow::error::ArrowError;
use std::sync::Arc;

/// Creates a schema for storing raster data in Apache Arrow format.
/// Utilizing nested structs and lists to represent raster metadata and bands.
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
            // Optional coordinate reference system (CRS) as json
            Field::new(column::CRS, DataType::Utf8View, true),
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
            Field::new(column::NODATAVALUE, DataType::Binary, true), // Allow null nodata values
            Field::new(column::STORAGE_TYPE, DataType::UInt32, false),
            Field::new(column::DATATYPE, DataType::UInt32, false),
        ]))
    }

    /// Band data schema (single binary blob)
    pub fn band_data_type() -> DataType {
        DataType::Binary
    }
}

#[repr(u16)]
#[derive(Clone, Debug, PartialEq, Eq)]
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
///   - Self-contained, no external dependencies, fast access for small-medium rasters
///   - Increases Arrow array size, memory usage grows and copy times increase with raster size
///   - Best for: Tiles, thumbnails, processed results, small rasters (<10MB per band)
///
/// **OutDbRef**: Raster data is stored externally with references in the Arrow array.
///   - Keeps Arrow arrays lightweight, supports massive rasters, enables lazy loading
///   - Requires external storage management, potential for broken references
///   - Best for: Large satellite imagery, time series data, cloud-native workflows
///   - Supported backends: S3, GCS, Azure Blob, local filesystem, HTTP endpoints
#[repr(u16)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StorageType {
    InDb = 0,
    OutDbRef = 1,
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

        let band_struct_builder = StructBuilder::from_fields(
            match RasterSchema::band_type() {
                DataType::Struct(fields) => fields,
                _ => panic!("Expected struct type for band"),
            },
            0, // Initial capacity for bands
        );

        let bands_builder = ListBuilder::new(band_struct_builder).with_field(Field::new(
            column::BAND,
            RasterSchema::band_type(),
            false,
        ));

        Self {
            metadata_builder,
            bands_builder,
        }
    }

    /// Start a new raster and write its metadata
    ///
    /// Accepts any type that implements MetadataRef, allowing you to pass:
    /// - RasterMetadata structs directly
    /// - MetadataRef trait objects from iterators
    pub fn start_raster(&mut self, metadata: &dyn MetadataRef) -> Result<(), ArrowError> {
        self.append_metadata_from_ref(metadata)
    }

    /// Convenience method for starting a raster with owned RasterMetadata
    pub fn start_raster_owned(&mut self, metadata: RasterMetadata) -> Result<(), ArrowError> {
        self.start_raster(&metadata)
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
                .field_builder::<BinaryBuilder>(band_metadata_indices::NODATAVALUE)
                .unwrap()
                .append_value(&nodata);
        } else {
            metadata_builder
                .field_builder::<BinaryBuilder>(band_metadata_indices::NODATAVALUE)
                .unwrap()
                .append_null();
        }

        metadata_builder
            .field_builder::<arrow::array::UInt32Builder>(band_metadata_indices::STORAGE_TYPE)
            .unwrap()
            .append_value(band_metadata.storage_type as u32);

        metadata_builder
            .field_builder::<arrow::array::UInt32Builder>(band_metadata_indices::DATATYPE)
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

    /// Append raster metadata from a MetadataRef trait object
    fn append_metadata_from_ref(&mut self, metadata: &dyn MetadataRef) -> Result<(), ArrowError> {
        // Width
        self.metadata_builder
            .field_builder::<arrow::array::UInt64Builder>(metadata_indices::WIDTH)
            .unwrap()
            .append_value(metadata.width());

        // Height
        self.metadata_builder
            .field_builder::<arrow::array::UInt64Builder>(metadata_indices::HEIGHT)
            .unwrap()
            .append_value(metadata.height());

        // Geotransform parameters
        self.metadata_builder
            .field_builder::<arrow::array::Float64Builder>(metadata_indices::UPPERLEFT_X)
            .unwrap()
            .append_value(metadata.upper_left_x());

        self.metadata_builder
            .field_builder::<arrow::array::Float64Builder>(metadata_indices::UPPERLEFT_Y)
            .unwrap()
            .append_value(metadata.upper_left_y());

        self.metadata_builder
            .field_builder::<arrow::array::Float64Builder>(metadata_indices::SCALE_X)
            .unwrap()
            .append_value(metadata.scale_x());

        self.metadata_builder
            .field_builder::<arrow::array::Float64Builder>(metadata_indices::SCALE_Y)
            .unwrap()
            .append_value(metadata.scale_y());

        self.metadata_builder
            .field_builder::<arrow::array::Float64Builder>(metadata_indices::SKEW_X)
            .unwrap()
            .append_value(metadata.skew_x());

        self.metadata_builder
            .field_builder::<arrow::array::Float64Builder>(metadata_indices::SKEW_Y)
            .unwrap()
            .append_value(metadata.skew_y());

        // Optional bounding box
        if let Some(bbox) = metadata.bounding_box() {
            let bbox_builder = self
                .metadata_builder
                .field_builder::<StructBuilder>(metadata_indices::BOUNDING_BOX)
                .unwrap();

            bbox_builder
                .field_builder::<arrow::array::Float64Builder>(bounding_box_indices::MIN_X)
                .unwrap()
                .append_value(bbox.min_x);

            bbox_builder
                .field_builder::<arrow::array::Float64Builder>(bounding_box_indices::MIN_Y)
                .unwrap()
                .append_value(bbox.min_y);

            bbox_builder
                .field_builder::<arrow::array::Float64Builder>(bounding_box_indices::MAX_X)
                .unwrap()
                .append_value(bbox.max_x);

            bbox_builder
                .field_builder::<arrow::array::Float64Builder>(bounding_box_indices::MAX_Y)
                .unwrap()
                .append_value(bbox.max_y);

            bbox_builder.append(true);
        } else {
            // Append null bounding box - need to fill in null values for all fields
            let bbox_builder = self
                .metadata_builder
                .field_builder::<StructBuilder>(metadata_indices::BOUNDING_BOX)
                .unwrap();

            bbox_builder
                .field_builder::<arrow::array::Float64Builder>(bounding_box_indices::MIN_X)
                .unwrap()
                .append_null();

            bbox_builder
                .field_builder::<arrow::array::Float64Builder>(bounding_box_indices::MIN_Y)
                .unwrap()
                .append_null();

            bbox_builder
                .field_builder::<arrow::array::Float64Builder>(bounding_box_indices::MAX_X)
                .unwrap()
                .append_null();

            bbox_builder
                .field_builder::<arrow::array::Float64Builder>(bounding_box_indices::MAX_Y)
                .unwrap()
                .append_null();

            bbox_builder.append(false);
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
        self.start_raster(&metadata)?;

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

/// Iterator and accessor traits for reading raster data from Arrow arrays.
///
/// These traits provide a zero-copy interface for accessing raster metadata and band data
/// from the Arrow-based storage format. The implementation handles both InDb and OutDbRef
/// storage types seamlessly.

/// Trait for accessing raster metadata (dimensions, geotransform, bounding box, etc.)
pub trait MetadataRef {
    /// Width of the raster in pixels (using u64 to match schema)
    fn width(&self) -> u64;
    /// Height of the raster in pixels (using u64 to match schema)  
    fn height(&self) -> u64;
    /// X coordinate of the upper-left corner
    fn upper_left_x(&self) -> f64;
    /// Y coordinate of the upper-left corner
    fn upper_left_y(&self) -> f64;
    /// X-direction pixel size (scale)
    fn scale_x(&self) -> f64;
    /// Y-direction pixel size (scale)
    fn scale_y(&self) -> f64;
    /// X-direction skew/rotation
    fn skew_x(&self) -> f64;
    /// Y-direction skew/rotation
    fn skew_y(&self) -> f64;
    /// Optional bounding box (when available)
    fn bounding_box(&self) -> Option<BoundingBox>;
    /// Optional coordinate reference system as binary data
    fn crs(&self) -> Option<&str>;
}

/// Implement MetadataRef for RasterMetadata to allow direct use with builder
impl MetadataRef for RasterMetadata {
    fn width(&self) -> u64 {
        self.width
    }
    fn height(&self) -> u64 {
        self.height
    }
    fn upper_left_x(&self) -> f64 {
        self.upperleft_x
    }
    fn upper_left_y(&self) -> f64 {
        self.upperleft_y
    }
    fn scale_x(&self) -> f64 {
        self.scale_x
    }
    fn scale_y(&self) -> f64 {
        self.scale_y
    }
    fn skew_x(&self) -> f64 {
        self.skew_x
    }
    fn skew_y(&self) -> f64 {
        self.skew_y
    }
    fn bounding_box(&self) -> Option<BoundingBox> {
        self.bounding_box.clone()
    }
    fn crs(&self) -> Option<&str> {
        self.crs.as_deref()
    }
}

/// Trait for accessing individual band metadata
pub trait BandMetadataRef {
    /// No-data value as raw bytes (None if null)
    fn nodata_value(&self) -> Option<&[u8]>;
    /// Storage type (InDb, OutDbRef, etc)
    fn storage_type(&self) -> StorageType;
    /// Band data type (UInt8, Float32, etc.)
    fn data_type(&self) -> BandDataType;
}

/// Trait for accessing individual band data
pub trait BandRef {
    /// Band metadata accessor
    fn metadata(&self) -> &dyn BandMetadataRef;
    /// Raw band data as bytes (zero-copy access)
    fn data(&self) -> &[u8];
}

/// Trait for accessing all bands in a raster
pub trait BandsRef {
    /// Number of bands in the raster
    fn len(&self) -> usize;
    /// Check if no bands are present
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Get a specific band by index (returns None if out of bounds)
    fn band(&self, index: usize) -> Option<Box<dyn BandRef + '_>>;
    /// Iterator over all bands
    fn iter(&self) -> BandIterator<'_>;
}

/// Trait for accessing complete raster data
pub trait RasterRef {
    /// Raster metadata accessor
    fn metadata(&self) -> &dyn MetadataRef;
    /// Bands accessor
    fn bands(&self) -> &dyn BandsRef;
}

/// Implementation of MetadataRef for Arrow StructArray
struct MetadataRefImpl<'a> {
    metadata_struct: &'a StructArray,
    index: usize,
}

impl<'a> MetadataRef for MetadataRefImpl<'a> {
    fn width(&self) -> u64 {
        self.metadata_struct
            .column(metadata_indices::WIDTH)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(self.index)
    }

    fn height(&self) -> u64 {
        self.metadata_struct
            .column(metadata_indices::HEIGHT)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(self.index)
    }

    fn upper_left_x(&self) -> f64 {
        self.metadata_struct
            .column(metadata_indices::UPPERLEFT_X)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap()
            .value(self.index)
    }

    fn upper_left_y(&self) -> f64 {
        self.metadata_struct
            .column(metadata_indices::UPPERLEFT_Y)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap()
            .value(self.index)
    }

    fn scale_x(&self) -> f64 {
        self.metadata_struct
            .column(metadata_indices::SCALE_X)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap()
            .value(self.index)
    }

    fn scale_y(&self) -> f64 {
        self.metadata_struct
            .column(metadata_indices::SCALE_Y)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap()
            .value(self.index)
    }

    fn skew_x(&self) -> f64 {
        self.metadata_struct
            .column(metadata_indices::SKEW_X)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap()
            .value(self.index)
    }

    fn skew_y(&self) -> f64 {
        self.metadata_struct
            .column(metadata_indices::SKEW_Y)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap()
            .value(self.index)
    }

    fn bounding_box(&self) -> Option<BoundingBox> {
        // Try to get bounding box if present in schema
        if let Some(bbox_struct) = self
            .metadata_struct
            .column(metadata_indices::BOUNDING_BOX)
            .as_any()
            .downcast_ref::<StructArray>()
        {
            Some(BoundingBox {
                min_x: bbox_struct
                    .column(bounding_box_indices::MIN_X)
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()?
                    .value(self.index),
                min_y: bbox_struct
                    .column(bounding_box_indices::MIN_Y)
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()?
                    .value(self.index),
                max_x: bbox_struct
                    .column(bounding_box_indices::MAX_X)
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()?
                    .value(self.index),
                max_y: bbox_struct
                    .column(bounding_box_indices::MAX_Y)
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()?
                    .value(self.index),
            })
        } else {
            None
        }
    }

    fn crs(&self) -> Option<&str> {
        let crs_array = self
            .metadata_struct
            .column(metadata_indices::CRS)
            .as_any()
            .downcast_ref::<StringView>()?;

        if crs_array.is_null(self.index) {
            None
        } else {
            Some(crs_array.value(self.index))
        }
    }
}

/// Implementation of BandMetadataRef for Arrow StructArray
struct BandMetadataRefImpl<'a> {
    metadata_struct: &'a StructArray,
    band_index: usize,
}

impl<'a> BandMetadataRef for BandMetadataRefImpl<'a> {
    fn nodata_value(&self) -> Option<&[u8]> {
        let nodata_array = self
            .metadata_struct
            .column(band_metadata_indices::NODATAVALUE)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .expect("Expected BinaryArray for nodata");

        if nodata_array.is_null(self.band_index) {
            None
        } else {
            Some(nodata_array.value(self.band_index))
        }
    }

    fn storage_type(&self) -> StorageType {
        let storage_type_array = self
            .metadata_struct
            .column(band_metadata_indices::STORAGE_TYPE)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("Expected UInt32Array for storage_type");

        match storage_type_array.value(self.band_index) {
            0 => StorageType::InDb,
            1 => StorageType::OutDbRef,
            _ => panic!(
                "Unknown storage type: {}",
                storage_type_array.value(self.band_index)
            ),
        }
    }

    fn data_type(&self) -> BandDataType {
        let datatype_array = self
            .metadata_struct
            .column(band_metadata_indices::DATATYPE)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("Expected UInt32Array for datatype");

        match datatype_array.value(self.band_index) {
            0 => BandDataType::UInt8,
            1 => BandDataType::UInt16,
            2 => BandDataType::Int16,
            3 => BandDataType::UInt32,
            4 => BandDataType::Int32,
            5 => BandDataType::Float32,
            6 => BandDataType::Float64,
            _ => panic!(
                "Unknown band data type: {}",
                datatype_array.value(self.band_index)
            ),
        }
    }
}

/// Implementation of BandRef for accessing individual band data
struct BandRefImpl<'a> {
    band_metadata: BandMetadataRefImpl<'a>,
    band_data: &'a [u8],
}

impl<'a> BandRef for BandRefImpl<'a> {
    fn metadata(&self) -> &dyn BandMetadataRef {
        &self.band_metadata
    }

    fn data(&self) -> &[u8] {
        self.band_data
    }
}

/// Implementation of BandsRef for accessing all bands in a raster  
struct BandsRefImpl<'a> {
    bands_list: &'a ListArray,
    raster_index: usize,
}

impl<'a> BandsRef for BandsRefImpl<'a> {
    fn len(&self) -> usize {
        let start = self.bands_list.value_offsets()[self.raster_index] as usize;
        let end = self.bands_list.value_offsets()[self.raster_index + 1] as usize;
        end - start
    }

    /// Get a specific band by index
    /// IMPORTANT: This function is utilizing zero based band indexing.
    ///            We may want to consider one-based indexing to match
    ///            raster standard band conventions.
    fn band(&self, index: usize) -> Option<Box<dyn BandRef + '_>> {
        if index >= self.len() {
            return None;
        }

        let start = self.bands_list.value_offsets()[self.raster_index] as usize;
        let band_row = start + index;

        let bands_struct = self
            .bands_list
            .values()
            .as_any()
            .downcast_ref::<StructArray>()?;

        // Get the metadata substructure from the band struct
        let band_metadata_struct = bands_struct
            .column(band_indices::METADATA)
            .as_any()
            .downcast_ref::<StructArray>()?;

        let band_metadata = BandMetadataRefImpl {
            metadata_struct: band_metadata_struct,
            band_index: band_row,
        };

        // Get band data from the Binary column within the band struct
        let band_data_array = bands_struct
            .column(band_indices::DATA)
            .as_any()
            .downcast_ref::<BinaryArray>()?;

        let band_data = band_data_array.value(band_row);

        Some(Box::new(BandRefImpl {
            band_metadata,
            band_data,
        }))
    }

    fn iter(&self) -> BandIterator<'_> {
        BandIterator {
            bands: self,
            current: 0,
        }
    }
}

/// Iterator for bands within a raster
pub struct BandIterator<'a> {
    bands: &'a dyn BandsRef,
    current: usize,
}

impl<'a> Iterator for BandIterator<'a> {
    type Item = Box<dyn BandRef + 'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.bands.len() {
            let band = self.bands.band(self.current);
            self.current += 1;
            band
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.bands.len().saturating_sub(self.current);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for BandIterator<'_> {}

/// Implementation of RasterRef for complete raster access
pub struct RasterRefImpl<'a> {
    metadata: MetadataRefImpl<'a>,
    bands: BandsRefImpl<'a>,
}

impl<'a> RasterRefImpl<'a> {
    /// Create a new RasterRefImpl from a struct array and index using hard-coded indices
    pub fn new(raster_struct: &'a StructArray, raster_index: usize) -> Self {
        let metadata_struct = raster_struct
            .column(raster_indices::METADATA)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();

        let bands_list = raster_struct
            .column(raster_indices::BANDS)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();

        let metadata = MetadataRefImpl {
            metadata_struct,
            index: raster_index,
        };

        let bands = BandsRefImpl {
            bands_list,
            raster_index,
        };

        Self { metadata, bands }
    }
}

impl<'a> RasterRef for RasterRefImpl<'a> {
    fn metadata(&self) -> &dyn MetadataRef {
        &self.metadata
    }

    fn bands(&self) -> &dyn BandsRef {
        &self.bands
    }
}

/// Iterator over raster structs in an Arrow StructArray
///
/// This provides efficient, zero-copy access to raster data stored in Arrow format.
/// Each iteration yields a `RasterRefImpl` that provides access to both metadata and band data.
pub struct RasterStructIterator<'a> {
    raster_array: &'a StructArray,
    current_row: usize,
}

impl<'a> RasterStructIterator<'a> {
    /// Create a new iterator over the raster struct array
    pub fn new(raster_array: &'a StructArray) -> Self {
        Self {
            raster_array,
            current_row: 0,
        }
    }

    /// Get the total number of rasters in the array
    pub fn len(&self) -> usize {
        self.raster_array.len()
    }

    /// Check if the array is empty
    pub fn is_empty(&self) -> bool {
        self.raster_array.is_empty()
    }

    /// Get a specific raster by index without consuming the iterator
    pub fn get(&self, index: usize) -> Option<RasterRefImpl<'a>> {
        if index >= self.raster_array.len() {
            return None;
        }

        Some(RasterRefImpl::new(self.raster_array, index))
    }
}

impl<'a> Iterator for RasterStructIterator<'a> {
    type Item = RasterRefImpl<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row < self.raster_array.len() {
            let result = self.get(self.current_row)?;
            self.current_row += 1;
            Some(result)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.raster_array.len().saturating_sub(self.current_row);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for RasterStructIterator<'_> {}

/// Convenience constructor function for creating a raster iterator
pub fn raster_iterator(raster_struct: &StructArray) -> RasterStructIterator<'_> {
    RasterStructIterator::new(raster_struct)
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
    pub crs: Option<&str>,
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

// Private field column name and index constants
// used across schema, builders and iterators
mod column {
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
    pub const CRS: &str = "crs";

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

/// Hard-coded column indices for maximum performance
/// These must match the exact order defined in RasterSchema::metadata_type()
mod metadata_indices {
    pub const WIDTH: usize = 0;
    pub const HEIGHT: usize = 1;
    pub const UPPERLEFT_X: usize = 2;
    pub const UPPERLEFT_Y: usize = 3;
    pub const SCALE_X: usize = 4;
    pub const SCALE_Y: usize = 5;
    pub const SKEW_X: usize = 6;
    pub const SKEW_Y: usize = 7;
    pub const BOUNDING_BOX: usize = 8;
    pub const CRS: usize = 9;
}

mod bounding_box_indices {
    pub const MIN_X: usize = 0;
    pub const MIN_Y: usize = 1;
    pub const MAX_X: usize = 2;
    pub const MAX_Y: usize = 3;
}

mod band_metadata_indices {
    pub const NODATAVALUE: usize = 0;
    pub const STORAGE_TYPE: usize = 1;
    pub const DATATYPE: usize = 2;
}

mod band_indices {
    pub const METADATA: usize = 0;
    pub const DATA: usize = 1;
}

mod raster_indices {
    pub const METADATA: usize = 0;
    pub const BANDS: usize = 1;
}

#[cfg(test)]
mod iterator_tests {
    use super::*;

    #[test]
    fn test_iterator_basic_functionality() {
        // Create a simple raster for testing using the correct API
        let mut builder = RasterBuilder::new(10); // capacity

        let metadata = RasterMetadata {
            width: 10,
            height: 10,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
            bounding_box: Some(BoundingBox {
                min_x: 0.0,
                min_y: -10.0,
                max_x: 10.0,
                max_y: 0.0,
            }),
            crs: None,
        };

        builder.start_raster(&metadata).unwrap();

        let band_metadata = BandMetadata {
            nodata_value: Some(vec![255u8]),
            storage_type: StorageType::InDb,
            datatype: BandDataType::UInt8,
        };

        // Add a single band with some test data using the correct API
        let test_data = vec![1u8; 100]; // 10x10 raster with value 1
        builder.band_data_writer().append_value(&test_data);
        builder.finish_band(band_metadata).unwrap();
        builder.finish_raster();

        let raster_array = builder.finish().unwrap();

        // Test the iterator
        let mut iterator = raster_iterator(&raster_array);

        assert_eq!(iterator.len(), 1);
        assert!(!iterator.is_empty());

        let raster = iterator.next().unwrap();
        let metadata = raster.metadata();

        assert_eq!(metadata.width(), 10);
        assert_eq!(metadata.height(), 10);
        assert_eq!(metadata.scale_x(), 1.0);
        assert_eq!(metadata.scale_y(), -1.0);

        let bbox = metadata.bounding_box().unwrap();
        assert_eq!(bbox.min_x, 0.0);
        assert_eq!(bbox.max_x, 10.0);

        let bands = raster.bands();
        assert_eq!(bands.len(), 1);
        assert!(!bands.is_empty());

        let band = bands.band(0).unwrap();
        assert_eq!(band.data().len(), 100);
        assert_eq!(band.data()[0], 1u8);

        let band_meta = band.metadata();
        assert_eq!(band_meta.storage_type(), StorageType::InDb);
        assert_eq!(band_meta.data_type(), BandDataType::UInt8);

        // Test iterator over bands
        let band_iter: Vec<_> = bands.iter().collect();
        assert_eq!(band_iter.len(), 1);
    }

    #[test]
    fn test_multi_band_iterator() {
        let mut builder = RasterBuilder::new(10);

        let metadata = RasterMetadata {
            width: 5,
            height: 5,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
            bounding_box: None,
            crs: None,
        };

        builder.start_raster(&metadata).unwrap();

        // Add three bands using the correct API
        for band_idx in 0..3 {
            let band_metadata = BandMetadata {
                nodata_value: Some(vec![255u8]),
                storage_type: StorageType::InDb,
                datatype: BandDataType::UInt8,
            };

            let test_data = vec![band_idx as u8; 25]; // 5x5 raster
            builder.band_data_writer().append_value(&test_data);
            builder.finish_band(band_metadata).unwrap();
        }

        builder.finish_raster();
        let raster_array = builder.finish().unwrap();

        let mut iterator = raster_iterator(&raster_array);
        let raster = iterator.next().unwrap();
        let bands = raster.bands();

        assert_eq!(bands.len(), 3);

        // Test each band has different data
        for i in 0..3 {
            let band = bands.band(i).unwrap();
            let expected_value = i as u8;
            assert!(band.data().iter().all(|&x| x == expected_value));
        }

        // Test iterator
        let band_values: Vec<u8> = bands
            .iter()
            .enumerate()
            .map(|(i, band)| {
                assert_eq!(band.data()[0], i as u8);
                band.data()[0]
            })
            .collect();

        assert_eq!(band_values, vec![0, 1, 2]);
    }

    #[test]
    fn test_copy_metadata_from_iterator() {
        // Create an original raster
        let mut source_builder = RasterBuilder::new(10);

        let original_metadata = RasterMetadata {
            width: 42,
            height: 24,
            upperleft_x: -122.0,
            upperleft_y: 37.8,
            scale_x: 0.1,
            scale_y: -0.1,
            skew_x: 0.0,
            skew_y: 0.0,
            bounding_box: Some(BoundingBox {
                min_x: -122.0,
                min_y: 35.4,
                max_x: -120.0,
                max_y: 37.8,
            }),
            crs: Some(b"EPSG:4326".to_vec()),
        };

        source_builder.start_raster(&original_metadata).unwrap();

        let band_metadata = BandMetadata {
            nodata_value: Some(vec![255u8]),
            storage_type: StorageType::InDb,
            datatype: BandDataType::UInt8,
        };

        let test_data = vec![42u8; 1008]; // 42x24 raster
        source_builder.band_data_writer().append_value(&test_data);
        source_builder.finish_band(band_metadata).unwrap();
        source_builder.finish_raster().unwrap();

        let source_array = source_builder.finish().unwrap();

        // Now create a new raster using metadata from the iterator - this is the key feature!
        let mut target_builder = RasterBuilder::new(10);
        let iterator = raster_iterator(&source_array);
        let source_raster = iterator.get(0).unwrap();

        // Use metadata directly from the iterator (zero-copy!)
        target_builder
            .start_raster(source_raster.metadata())
            .unwrap();

        // Add new band data while preserving original metadata
        let new_band_metadata = BandMetadata {
            nodata_value: None,
            storage_type: StorageType::InDb,
            datatype: BandDataType::UInt16,
        };

        let new_data = vec![100u16; 1008]; // Different data, same dimensions
        let new_data_bytes: Vec<u8> = new_data.iter().flat_map(|&x| x.to_le_bytes()).collect();

        target_builder
            .band_data_writer()
            .append_value(&new_data_bytes);
        target_builder.finish_band(new_band_metadata).unwrap();
        target_builder.finish_raster().unwrap();

        let target_array = target_builder.finish().unwrap();

        // Verify the metadata was copied correctly
        let target_iterator = raster_iterator(&target_array);
        let target_raster = target_iterator.get(0).unwrap();
        let target_metadata = target_raster.metadata();

        // All metadata should match the original
        assert_eq!(target_metadata.width(), 42);
        assert_eq!(target_metadata.height(), 24);
        assert_eq!(target_metadata.upper_left_x(), -122.0);
        assert_eq!(target_metadata.upper_left_y(), 37.8);
        assert_eq!(target_metadata.scale_x(), 0.1);
        assert_eq!(target_metadata.scale_y(), -0.1);

        let target_bbox = target_metadata.bounding_box().unwrap();
        assert_eq!(target_bbox.min_x, -122.0);
        assert_eq!(target_bbox.max_x, -120.0);

        // But band data and metadata should be different
        let target_band = target_raster.bands().band(0).unwrap();
        let target_band_meta = target_band.metadata();
        assert_eq!(target_band_meta.data_type(), BandDataType::UInt16);
        assert!(target_band_meta.nodata_value().is_none());
        assert_eq!(target_band.data().len(), 2016); // 1008 * 2 bytes per u16
    }

    #[test]
    fn test_random_access() {
        let mut builder = RasterBuilder::new(10);

        // Add multiple rasters
        for raster_idx in 0..3 {
            let metadata = RasterMetadata {
                width: raster_idx as u64 + 1,
                height: raster_idx as u64 + 1,
                upperleft_x: raster_idx as f64,
                upperleft_y: raster_idx as f64,
                scale_x: 1.0,
                scale_y: -1.0,
                skew_x: 0.0,
                skew_y: 0.0,
                bounding_box: None,
                crs: None,
            };

            builder.start_raster(&metadata).unwrap();

            let band_metadata = BandMetadata {
                nodata_value: Some(vec![255u8]),
                storage_type: StorageType::InDb,
                datatype: BandDataType::UInt8,
            };

            let size = (raster_idx + 1) * (raster_idx + 1);
            let test_data = vec![raster_idx as u8; size];
            builder.band_data_writer().append_value(&test_data);
            builder.finish_band(band_metadata).unwrap();
            builder.finish_raster();
        }

        let raster_array = builder.finish().unwrap();
        let iterator = raster_iterator(&raster_array);

        assert_eq!(iterator.len(), 3);

        // Test random access
        let raster_2 = iterator.get(2).unwrap();
        assert_eq!(raster_2.metadata().width(), 3);
        assert_eq!(raster_2.metadata().height(), 3);
        assert_eq!(raster_2.metadata().upper_left_x(), 2.0);

        let band = raster_2.bands().band(0).unwrap();
        assert_eq!(band.data().len(), 9);
        assert!(band.data().iter().all(|&x| x == 2u8));

        // Test out of bounds
        assert!(iterator.get(10).is_none());
    }

    /// Comprehensive test to verify all hard-coded indices match the actual schema
    #[test]
    fn test_hardcoded_indices_match_schema() {
        // Test raster-level indices
        let raster_fields = RasterSchema::fields();
        assert_eq!(raster_fields.len(), 2, "Expected exactly 2 raster fields");
        assert_eq!(
            raster_fields[raster_indices::METADATA].name(),
            column::METADATA,
            "Raster metadata index mismatch"
        );
        assert_eq!(
            raster_fields[raster_indices::BANDS].name(),
            column::BANDS,
            "Raster bands index mismatch"
        );

        // Test metadata indices
        let metadata_type = RasterSchema::metadata_type();
        if let DataType::Struct(metadata_fields) = metadata_type {
            assert_eq!(
                metadata_fields.len(),
                10,
                "Expected exactly 10 metadata fields"
            );
            assert_eq!(
                metadata_fields[metadata_indices::WIDTH].name(),
                column::WIDTH,
                "Metadata width index mismatch"
            );
            assert_eq!(
                metadata_fields[metadata_indices::HEIGHT].name(),
                column::HEIGHT,
                "Metadata height index mismatch"
            );
            assert_eq!(
                metadata_fields[metadata_indices::UPPERLEFT_X].name(),
                column::UPPERLEFT_X,
                "Metadata upperleft_x index mismatch"
            );
            assert_eq!(
                metadata_fields[metadata_indices::UPPERLEFT_Y].name(),
                column::UPPERLEFT_Y,
                "Metadata upperleft_y index mismatch"
            );
            assert_eq!(
                metadata_fields[metadata_indices::SCALE_X].name(),
                column::SCALE_X,
                "Metadata scale_x index mismatch"
            );
            assert_eq!(
                metadata_fields[metadata_indices::SCALE_Y].name(),
                column::SCALE_Y,
                "Metadata scale_y index mismatch"
            );
            assert_eq!(
                metadata_fields[metadata_indices::SKEW_X].name(),
                column::SKEW_X,
                "Metadata skew_x index mismatch"
            );
            assert_eq!(
                metadata_fields[metadata_indices::SKEW_Y].name(),
                column::SKEW_Y,
                "Metadata skew_y index mismatch"
            );
            assert_eq!(
                metadata_fields[metadata_indices::BOUNDING_BOX].name(),
                column::BOUNDING_BOX,
                "Metadata bounding_box index mismatch"
            );
            assert_eq!(
                metadata_fields[metadata_indices::CRS].name(),
                column::CRS,
                "Metadata crs index mismatch"
            );
        } else {
            panic!("Expected Struct type for metadata");
        }

        // Test bounding box indices
        let bbox_type = RasterSchema::bounding_box_type();
        if let DataType::Struct(bbox_fields) = bbox_type {
            assert_eq!(
                bbox_fields.len(),
                4,
                "Expected exactly 4 bounding box fields"
            );
            assert_eq!(
                bbox_fields[bounding_box_indices::MIN_X].name(),
                column::MIN_X,
                "Bounding box min_x index mismatch"
            );
            assert_eq!(
                bbox_fields[bounding_box_indices::MIN_Y].name(),
                column::MIN_Y,
                "Bounding box min_y index mismatch"
            );
            assert_eq!(
                bbox_fields[bounding_box_indices::MAX_X].name(),
                column::MAX_X,
                "Bounding box max_x index mismatch"
            );
            assert_eq!(
                bbox_fields[bounding_box_indices::MAX_Y].name(),
                column::MAX_Y,
                "Bounding box max_y index mismatch"
            );
        } else {
            panic!("Expected Struct type for bounding box");
        }

        // Test band metadata indices
        let band_metadata_type = RasterSchema::band_metadata_type();
        if let DataType::Struct(band_metadata_fields) = band_metadata_type {
            assert_eq!(
                band_metadata_fields.len(),
                3,
                "Expected exactly 3 band metadata fields"
            );
            assert_eq!(
                band_metadata_fields[band_metadata_indices::NODATAVALUE].name(),
                column::NODATAVALUE,
                "Band metadata nodatavalue index mismatch"
            );
            assert_eq!(
                band_metadata_fields[band_metadata_indices::STORAGE_TYPE].name(),
                column::STORAGE_TYPE,
                "Band metadata storage_type index mismatch"
            );
            assert_eq!(
                band_metadata_fields[band_metadata_indices::DATATYPE].name(),
                column::DATATYPE,
                "Band metadata datatype index mismatch"
            );
        } else {
            panic!("Expected Struct type for band metadata");
        }

        // Test band indices
        let band_type = RasterSchema::band_type();
        if let DataType::Struct(band_fields) = band_type {
            assert_eq!(band_fields.len(), 2, "Expected exactly 2 band fields");
            assert_eq!(
                band_fields[band_indices::METADATA].name(),
                column::METADATA,
                "Band metadata index mismatch"
            );
            assert_eq!(
                band_fields[band_indices::DATA].name(),
                column::DATA,
                "Band data index mismatch"
            );
        } else {
            panic!("Expected Struct type for band");
        }
    }
}
