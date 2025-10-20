// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
use arrow::buffer::{MutableBuffer, BooleanBuffer, NullBuffer};
use arrow_array::{
    builder::{
        BinaryBuilder, Float64Builder, ListBuilder, StringBuilder, StructBuilder,
        UInt32Builder, UInt64Builder,
    },
    Array, ArrayRef, BinaryArray, Float64Array, ListArray, StringArray, StringViewArray, StructArray,
    UInt32Array, UInt64Array,
};
use arrow_schema::{ArrowError, DataType, Field, FieldRef, Fields};
use datafusion_common::error::{DataFusionError, Result};
use sedona_common::sedona_internal_err;
use serde_json::Value;
use std::fmt::{Debug, Display};
use std::sync::{Arc, LazyLock};

use crate::crs::{deserialize_crs, Crs};
use crate::extension_type::ExtensionType;

/// Data types supported by Sedona that resolve to a concrete Arrow DataType
#[derive(Debug, PartialEq, Clone)]
pub enum SedonaType {
    Arrow(DataType),
    Wkb(Edges, Crs),
    WkbView(Edges, Crs),
    Raster(RasterSchema),
}

impl From<DataType> for SedonaType {
    fn from(value: DataType) -> Self {
        Self::Arrow(value)
    }
}

/// Edge interpolations
///
/// While at the logical level we refer to geometries and geographies, at the execution
/// layer we can reuse implementations for structural manipulation more efficiently if
/// we consider the edge interpolation as a parameter of the physical type. This maps to
/// the concept of "edges" in GeoArrow and "algorithm" in Parquet and Iceberg (where the
/// planar case would be resolved to a geometry instead of a geography).
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Edges {
    Planar,
    Spherical,
}

/// Sentinel for [`SedonaType::Wkb`] with planar edges
///
/// This constant is useful when defining type signatures as these ignore the Crs when
/// matching (and `SedonaType::Wkb(...)` is verbose)
pub const WKB_GEOMETRY: SedonaType = SedonaType::Wkb(Edges::Planar, Crs::None);

/// Sentinel for [`SedonaType::WkbView`] with planar edges
///
/// See [`WKB_GEOMETRY`]
pub const WKB_VIEW_GEOMETRY: SedonaType = SedonaType::WkbView(Edges::Planar, Crs::None);

/// Sentinel for [`SedonaType::Wkb`] with spherical edges
///
/// This constant is useful when defining type signatures as these ignore the Crs when
/// matching (and `SedonaType::Wkb(...)` is verbose)
pub const WKB_GEOGRAPHY: SedonaType = SedonaType::Wkb(Edges::Spherical, Crs::None);

/// Sentinel for [`SedonaType::WkbView`] with spherical edges
///
/// See [`WKB_GEOGRAPHY`]
pub const WKB_VIEW_GEOGRAPHY: SedonaType = SedonaType::WkbView(Edges::Spherical, Crs::None);

/// Sentinel for [`Sedona::RasterSchema`]
///
/// The CRS is stored within the raster schema.
pub const RASTER: SedonaType = SedonaType::Raster(RasterSchema);

/// Create a static value for the [`SedonaType::Raster`] that's initialized exactly once,
/// on first access
static RASTER_DATATYPE: LazyLock<DataType> =
    LazyLock::new(|| DataType::Struct(RasterSchema::fields()));

impl SedonaType {
    /// Given a field as it would appear in an external Schema return the appropriate SedonaType
    pub fn from_storage_field(field: &Field) -> Result<SedonaType> {
        match ExtensionType::from_field(field) {
            Some(ext) => Self::from_extension_type(ext),
            None => Ok(Self::Arrow(field.data_type().clone())),
        }
    }

    /// Given an [`ExtensionType`], construct a SedonaType
    pub fn from_extension_type(extension: ExtensionType) -> Result<SedonaType> {
        let (edges, crs) = deserialize_edges_and_crs(&extension.extension_metadata)?;
        if extension.extension_name == "geoarrow.wkb" {
            sedona_type_wkb(edges, crs, extension.storage_type)
        } else {
            sedona_internal_err!(
                "Extension type not implemented: <{}>:{}",
                extension.extension_name,
                extension.storage_type
            )
        }
    }

    /// Construct a [`Field`] as it would appear in an external `RecordBatch`
    pub fn to_storage_field(&self, name: &str, nullable: bool) -> Result<Field> {
        self.extension_type().map_or(
            Ok(Field::new(name, self.storage_type().clone(), nullable)),
            |extension| Ok(extension.to_field(name, nullable)),
        )
    }

    /// Compute the storage [`DataType`] as it would appear in an external `RecordBatch`
    pub fn storage_type(&self) -> &DataType {
        match self {
            SedonaType::Arrow(data_type) => data_type,
            SedonaType::Wkb(_, _) => &DataType::Binary,
            SedonaType::WkbView(_, _) => &DataType::BinaryView,
            SedonaType::Raster(_) => &RASTER_DATATYPE,
        }
    }

    /// Compute the extension name if this is an Arrow extension type or `None` otherwise
    pub fn extension_name(&self) -> Option<&'static str> {
        match self {
            SedonaType::Arrow(_) => None,
            SedonaType::Wkb(_, _) | SedonaType::WkbView(_, _) => Some("geoarrow.wkb"),
            SedonaType::Raster(_) => Some("sedona.raster"),
        }
    }

    /// Construct the [`ExtensionType`] that represents this type, if any
    pub fn extension_type(&self) -> Option<ExtensionType> {
        match self {
            SedonaType::Wkb(edges, crs) | SedonaType::WkbView(edges, crs) => {
                Some(ExtensionType::new(
                    self.extension_name().unwrap(),
                    self.storage_type().clone(),
                    Some(serialize_edges_and_crs(edges, crs)),
                ))
            }
            SedonaType::Raster(_) => Some(ExtensionType::new(
                self.extension_name().unwrap(),
                self.storage_type().clone(),
                None,
            )),
            _ => None,
        }
    }

    /// The logical type name for this type
    ///
    /// The logical type name is used in tabular display and schema printing. Notably,
    /// it renders Wkb and WkbView as "geometry" or "geography" depending on the edge
    /// type. For Arrow types, this similarly strips the storage details (e.g.,
    /// both Utf8 and Utf8View types render as "utf8").
    pub fn logical_type_name(&self) -> String {
        match self {
            SedonaType::Wkb(Edges::Planar, _) | SedonaType::WkbView(Edges::Planar, _) => {
                "geometry".to_string()
            }
            SedonaType::Wkb(Edges::Spherical, _) | SedonaType::WkbView(Edges::Spherical, _) => {
                "geography".to_string()
            }
            SedonaType::Arrow(data_type) => match data_type {
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => "utf8".to_string(),
                DataType::Binary
                | DataType::LargeBinary
                | DataType::BinaryView
                | DataType::FixedSizeBinary(_) => "binary".to_string(),
                DataType::List(_)
                | DataType::LargeList(_)
                | DataType::ListView(_)
                | DataType::LargeListView(_)
                | DataType::FixedSizeList(_, _) => "list".to_string(),
                DataType::Dictionary(_, value_type) => {
                    SedonaType::Arrow(value_type.as_ref().clone()).logical_type_name()
                }
                DataType::RunEndEncoded(_, value_field) => {
                    match SedonaType::from_storage_field(value_field) {
                        Ok(value_sedona_type) => value_sedona_type.logical_type_name(),
                        Err(_) => format!("{value_field:?}"),
                    }
                }
                _ => {
                    let data_type_str = data_type.to_string();
                    if let Some(params_start) = data_type_str.find('(') {
                        data_type_str[0..params_start].to_string().to_lowercase()
                    } else {
                        data_type_str.to_lowercase()
                    }
                }
            },
            SedonaType::Raster(_) => "raster".to_string(),
        }
    }

    /// Returns True if another physical type matches this one for the purposes of dispatch
    ///
    /// For Arrow types this matches on type equality; for other type it matches on edges
    /// but not crs.
    pub fn match_signature(&self, other: &SedonaType) -> bool {
        match (self, other) {
            (SedonaType::Arrow(data_type), SedonaType::Arrow(other_data_type)) => {
                data_type == other_data_type
            }
            (SedonaType::Wkb(edges, _), SedonaType::Wkb(other_edges, _)) => edges == other_edges,
            (SedonaType::WkbView(edges, _), SedonaType::WkbView(other_edges, _)) => {
                edges == other_edges
            }
            (SedonaType::Raster(_), SedonaType::Raster(_)) => true,
            _ => false,
        }
    }
}

// Implementation details for type serialization and display

impl Display for SedonaType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SedonaType::Arrow(data_type) => Display::fmt(data_type, f),
            SedonaType::Wkb(edges, crs) => display_geometry("Wkb", edges, crs, f),
            SedonaType::WkbView(edges, crs) => display_geometry("WkbView", edges, crs, f),
            SedonaType::Raster(_) => write!(f, "Raster"),
        }
    }
}

fn display_geometry(
    name: &str,
    edges: &Edges,
    crs: &Crs,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    let mut params = Vec::new();

    if let Some(crs) = crs {
        params.push(crs.to_string());
    }

    match edges {
        Edges::Planar => {}
        Edges::Spherical => {
            params.push("Spherical".to_string());
        }
    }

    match params.len() {
        0 => write!(f, "{name}")?,
        1 => write!(f, "{name}({})", params[0])?,
        _ => write!(f, "{name}({})", params.join(", "))?,
    }

    Ok(())
}

// Implementation details for importing/exporting types from/to Arrow + metadata

/// Check a storage type for SedonaType::Wkb
fn sedona_type_wkb(edges: Edges, crs: Crs, storage_type: DataType) -> Result<SedonaType> {
    match storage_type {
        DataType::Binary => Ok(SedonaType::Wkb(edges, crs)),
        DataType::BinaryView => Ok(SedonaType::WkbView(edges, crs)),
        _ => sedona_internal_err!(
            "Expected Wkb type with Binary storage but got {}",
            storage_type
        ),
    }
}

/// Parse a GeoArrow metadata string
///
/// Deserializes the extension metadata from a GeoArrow extension type. See
/// https://geoarrow.org/extension-types.html for a full definition of the metadata
/// format.
fn deserialize_edges_and_crs(value: &Option<String>) -> Result<(Edges, Crs)> {
    match value {
        Some(val) => {
            if val.is_empty() || val == "{}" {
                return Ok((Edges::Planar, Crs::None));
            }

            let json_value: Value = serde_json::from_str(val).map_err(|err| {
                DataFusionError::Internal(format!("Error deserializing GeoArrow metadata: {err}"))
            })?;
            if !json_value.is_object() {
                return sedona_internal_err!(
                    "Expected GeoArrow metadata as JSON object but got {}",
                    val
                );
            }

            let edges = match json_value.get("edges") {
                Some(edges_value) => deserialize_edges(edges_value)?,
                None => Edges::Planar,
            };

            let crs = match json_value.get("crs") {
                Some(crs_value) => deserialize_crs(crs_value)?,
                None => Crs::None,
            };

            Ok((edges, crs))
        }
        None => Ok((Edges::Planar, Crs::None)),
    }
}

/// Create a GeoArrow metadata string
///
/// Deserializes the extension metadata from a GeoArrow extension type. See
/// https://geoarrow.org/extension-types.html for a full definition of the metadata
/// format.
fn serialize_edges_and_crs(edges: &Edges, crs: &Crs) -> String {
    let crs_component = crs
        .as_ref()
        .map(|crs| format!(r#""crs":{}"#, crs.to_json()));

    let edges_component = match edges {
        Edges::Planar => None,
        Edges::Spherical => Some(r#""edges":"spherical""#),
    };

    match (crs_component, edges_component) {
        (None, None) => "{}".to_string(),
        (None, Some(edges)) => format!("{{{edges}}}"),
        (Some(crs), None) => format!("{{{crs}}}"),
        (Some(crs), Some(edges)) => format!("{{{edges},{crs}}}"),
    }
}

/// Deserialize a specific GeoArrow "edges" value
fn deserialize_edges(edges: &Value) -> Result<Edges> {
    match edges.as_str() {
        Some(edges_str) => {
            if edges_str == "planar" {
                Ok(Edges::Planar)
            } else if edges_str == "spherical" {
                Ok(Edges::Spherical)
            } else {
                sedona_internal_err!("Unsupported edges value {}", edges_str)
            }
        }
        None => {
            sedona_internal_err!("Unsupported edges JSON type in metadata {}", edges)
        }
    }
}

/// Schema for storing raster data in Apache Arrow format.
/// Utilizing nested structs and lists to represent raster metadata and bands.
#[derive(Debug, PartialEq, Clone)]
pub struct RasterSchema;
impl RasterSchema {
    // Raster schema:
    pub fn fields() -> Fields {
        Fields::from(vec![
            Field::new(column::METADATA, Self::metadata_type(), false),
            Field::new(column::CRS, Self::crs_type(), true),
            Field::new(column::BBOX, Self::bounding_box_type(), true),
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
            // OutDb reference fields - only used when storage_type == OutDbRef
            Field::new(column::OUTDB_URL, DataType::Utf8, true),
            Field::new(column::OUTDB_BAND_ID, DataType::UInt32, true),
        ]))
    }

    /// Band data schema (single binary blob)
    pub fn band_data_type() -> DataType {
        DataType::Binary // consider switching to BinaryView
    }

    /// CRS schema to store json representation
    pub fn crs_type() -> DataType {
        DataType::Utf8
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
    main_builder: StructBuilder,
}

impl RasterBuilder {
    /// Create a new raster builder with the specified capacity
    pub fn new(capacity: usize) -> Self {
        // Create individual builders that we know work
        let metadata_builder = StructBuilder::from_fields(
            match RasterSchema::metadata_type() {
                DataType::Struct(fields) => fields,
                _ => panic!("Expected struct type for metadata"),
            },
            capacity,
        );

        let crs_builder = StringBuilder::new();

        let bbox_builder = StructBuilder::from_fields(
            match RasterSchema::bounding_box_type() {
                DataType::Struct(fields) => fields,
                _ => panic!("Expected struct type for bounding box"),
            },
            capacity,
        );

        let band_struct_builder = StructBuilder::from_fields(
            match RasterSchema::band_type() {
                DataType::Struct(fields) => fields,
                _ => panic!("Expected struct type for band"),
            },
            0,
        );

        let bands_builder = ListBuilder::new(band_struct_builder).with_field(Field::new(
            column::BAND,
            RasterSchema::band_type(),
            false,
        ));

        // Now create the main builder with pre-built components
        let mut main_builder = StructBuilder::new(
            RasterSchema::fields(),
            vec![
                Box::new(metadata_builder),
                Box::new(crs_builder), 
                Box::new(bbox_builder),
                Box::new(bands_builder),
            ],
        );

        Self {
            main_builder,
        }
    }

    /// Start a new raster with metadata, optional CRS, and optional bounding box
    ///
    /// This is the unified method for starting a raster with all optional parameters.
    ///
    /// # Arguments
    /// * `metadata` - Raster metadata (dimensions, geotransform parameters)
    /// * `crs` - Optional coordinate reference system as string
    /// * `bbox` - Optional bounding box coordinates
    ///
    /// # Examples
    /// // From iterator - copy all fields from existing raster
    /// builder.start_raster(raster.metadata(), raster.crs(), raster.bounding_box(0).as_ref())?;
    ///
    /// // From RasterMetadata struct with all fields
    /// builder.start_raster(&metadata, Some("EPSG:4326"), metadata.bounding_box.as_ref())?;
    ///
    /// // Minimal - just metadata
    /// builder.start_raster(&metadata, None, None)?;
    /// ```
    pub fn start_raster(
        &mut self,
        metadata: &dyn MetadataRef,
        crs: Option<&str>,
        bbox: Option<&BoundingBox>,
    ) -> Result<(), ArrowError> {
        self.append_metadata_from_ref(metadata)?;
        self.set_crs(crs)?;
        self.append_bounding_box(bbox)?;
        Ok(())
    }

    /// Get direct access to the BinaryBuilder for writing the current band's data
    pub fn band_data_writer(&mut self) -> &mut BinaryBuilder {
        let bands_builder = self.main_builder
            .field_builder::<ListBuilder<StructBuilder>>(raster_indices::BANDS)
            .unwrap();
        let band_builder = bands_builder.values();
        // Ensure we have at least one field (band metadata and data)
        // Field 0 = metadata (StructBuilder), Field 1 = data (BinaryBuilder) 
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
        let bands_builder = self.main_builder
            .field_builder::<ListBuilder<StructBuilder>>(raster_indices::BANDS)
            .unwrap();
        let band_builder = bands_builder.values();

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
            .field_builder::<UInt32Builder>(band_metadata_indices::STORAGE_TYPE)
            .unwrap()
            .append_value(band_metadata.storage_type as u32);

        metadata_builder
            .field_builder::<UInt32Builder>(band_metadata_indices::DATATYPE)
            .unwrap()
            .append_value(band_metadata.datatype as u32);

        // Handle OutDb URL
        if let Some(url) = band_metadata.outdb_url {
            metadata_builder
                .field_builder::<StringBuilder>(band_metadata_indices::OUTDB_URL)
                .unwrap()
                .append_value(&url);
        } else {
            metadata_builder
                .field_builder::<StringBuilder>(band_metadata_indices::OUTDB_URL)
                .unwrap()
                .append_null();
        }

        // Handle OutDb band ID
        if let Some(band_id) = band_metadata.outdb_band_id {
            metadata_builder
                .field_builder::<UInt32Builder>(band_metadata_indices::OUTDB_BAND_ID)
                .unwrap()
                .append_value(band_id);
        } else {
            metadata_builder
                .field_builder::<UInt32Builder>(band_metadata_indices::OUTDB_BAND_ID)
                .unwrap()
                .append_null();
        }

        metadata_builder.append(true);

        // Finish the band
        band_builder.append(true);
        Ok(())
    }

    /// Finish all bands for the current raster
    pub fn finish_raster(&mut self) -> Result<(), ArrowError> {
        let bands_builder = self.main_builder
            .field_builder::<ListBuilder<StructBuilder>>(raster_indices::BANDS)
            .unwrap();
        bands_builder.append(true);
        // Mark this raster as valid (not null) in the main struct
        self.main_builder.append(true);
        Ok(())
    }

    /// Append raster metadata from a MetadataRef trait object
    fn append_metadata_from_ref(&mut self, metadata: &dyn MetadataRef) -> Result<(), ArrowError> {
        let metadata_builder = self.main_builder
            .field_builder::<StructBuilder>(raster_indices::METADATA)
            .unwrap();
            
        // Width
        metadata_builder
            .field_builder::<UInt64Builder>(metadata_indices::WIDTH)
            .unwrap()
            .append_value(metadata.width());

        // Height
        metadata_builder
            .field_builder::<UInt64Builder>(metadata_indices::HEIGHT)
            .unwrap()
            .append_value(metadata.height());

        // Geotransform parameters
        metadata_builder
            .field_builder::<Float64Builder>(metadata_indices::UPPERLEFT_X)
            .unwrap()
            .append_value(metadata.upper_left_x());

        metadata_builder
            .field_builder::<Float64Builder>(metadata_indices::UPPERLEFT_Y)
            .unwrap()
            .append_value(metadata.upper_left_y());

        metadata_builder
            .field_builder::<Float64Builder>(metadata_indices::SCALE_X)
            .unwrap()
            .append_value(metadata.scale_x());

        metadata_builder
            .field_builder::<Float64Builder>(metadata_indices::SCALE_Y)
            .unwrap()
            .append_value(metadata.scale_y());

        metadata_builder
            .field_builder::<Float64Builder>(metadata_indices::SKEW_X)
            .unwrap()
            .append_value(metadata.skew_x());

        metadata_builder
            .field_builder::<Float64Builder>(metadata_indices::SKEW_Y)
            .unwrap()
            .append_value(metadata.skew_y());

        metadata_builder.append(true);

        Ok(())
    }

    /// Set the CRS for the current raster
    pub fn set_crs(&mut self, crs: Option<&str>) -> Result<(), ArrowError> {
        let crs_builder = self.main_builder
            .field_builder::<StringBuilder>(raster_indices::CRS)
            .unwrap();
        match crs {
            Some(crs_data) => crs_builder.append_value(crs_data),
            None => crs_builder.append_null(),
        }
        Ok(())
    }

    /// Append a bounding box to the current raster
    pub fn append_bounding_box(&mut self, bbox: Option<&BoundingBox>) -> Result<(), ArrowError> {
        let bbox_builder = self.main_builder
            .field_builder::<StructBuilder>(raster_indices::BBOX)
            .unwrap();
            
        if let Some(bbox) = bbox {
            bbox_builder
                .field_builder::<Float64Builder>(bounding_box_indices::MIN_X)
                .unwrap()
                .append_value(bbox.min_x);

            bbox_builder
                .field_builder::<Float64Builder>(bounding_box_indices::MIN_Y)
                .unwrap()
                .append_value(bbox.min_y);

            bbox_builder
                .field_builder::<Float64Builder>(bounding_box_indices::MAX_X)
                .unwrap()
                .append_value(bbox.max_x);

            bbox_builder
                .field_builder::<Float64Builder>(bounding_box_indices::MAX_Y)
                .unwrap()
                .append_value(bbox.max_y);

            bbox_builder.append(true);
        } else {
            // Append null bounding box - need to fill in null values for all fields
            bbox_builder
                .field_builder::<Float64Builder>(bounding_box_indices::MIN_X)
                .unwrap()
                .append_null();

            bbox_builder
                .field_builder::<Float64Builder>(bounding_box_indices::MIN_Y)
                .unwrap()
                .append_null();

            bbox_builder
                .field_builder::<Float64Builder>(bounding_box_indices::MAX_X)
                .unwrap()
                .append_null();

            bbox_builder
                .field_builder::<Float64Builder>(bounding_box_indices::MAX_Y)
                .unwrap()
                .append_null();

            bbox_builder.append(false);
        }
        Ok(())
    }

    /// Append a null raster
    pub fn append_null(&mut self) -> Result<(), ArrowError> {
        // Since metadata fields are non-nullable, provide default values
        let metadata_builder = self.main_builder
            .field_builder::<StructBuilder>(raster_indices::METADATA)
            .unwrap();
            
        metadata_builder
            .field_builder::<UInt64Builder>(metadata_indices::WIDTH)
            .unwrap()
            .append_value(0u64);
        
        metadata_builder
            .field_builder::<UInt64Builder>(metadata_indices::HEIGHT)
            .unwrap()
            .append_value(0u64);
        
        metadata_builder
            .field_builder::<Float64Builder>(metadata_indices::UPPERLEFT_X)
            .unwrap()
            .append_value(0.0f64);
        
        metadata_builder
            .field_builder::<Float64Builder>(metadata_indices::UPPERLEFT_Y)
            .unwrap()
            .append_value(0.0f64);
        
        metadata_builder
            .field_builder::<Float64Builder>(metadata_indices::SCALE_X)
            .unwrap()
            .append_value(0.0f64);
        
        metadata_builder
            .field_builder::<Float64Builder>(metadata_indices::SCALE_Y)
            .unwrap()
            .append_value(0.0f64);
        
        metadata_builder
            .field_builder::<Float64Builder>(metadata_indices::SKEW_X)
            .unwrap()
            .append_value(0.0f64);
        
        metadata_builder
            .field_builder::<Float64Builder>(metadata_indices::SKEW_Y)
            .unwrap()
            .append_value(0.0f64);

        // Mark the metadata struct as valid since it has valid values
        metadata_builder.append(true);
        
        // Append null CRS (now using StringBuilder instead of StringViewBuilder)
        let crs_builder = self.main_builder
            .field_builder::<StringBuilder>(raster_indices::CRS)
            .unwrap();
        crs_builder.append_null();
        
        // Append null bounding box
        self.append_bounding_box(None)?;
        
        // Append null bands
        let bands_builder = self.main_builder
            .field_builder::<ListBuilder<StructBuilder>>(raster_indices::BANDS)
            .unwrap();
        bands_builder.append(false);

        // Mark this raster as null in the main struct
        self.main_builder.append(false);

        Ok(())
    }

    /// Finish building and return the constructed StructArray
    pub fn finish(mut self) -> Result<StructArray, ArrowError> {
        Ok(self.main_builder.finish())
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
        self.start_raster(&metadata, None, metadata.bounding_box.as_ref())?;

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
}

/// Trait for accessing individual band metadata
pub trait BandMetadataRef {
    /// No-data value as raw bytes (None if null)
    fn nodata_value(&self) -> Option<&[u8]>;
    /// Storage type (InDb, OutDbRef, etc)
    fn storage_type(&self) -> StorageType;
    /// Band data type (UInt8, Float32, etc.)
    fn data_type(&self) -> BandDataType;
    /// OutDb URL (only used when storage_type == OutDbRef)
    fn outdb_url(&self) -> Option<&str>;
    /// OutDb band ID (only used when storage_type == OutDbRef)
    fn outdb_band_id(&self) -> Option<u32>;
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
    /// CRS accessor
    fn crs(&self) -> Option<&str>;
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
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(self.index)
    }

    fn upper_left_y(&self) -> f64 {
        self.metadata_struct
            .column(metadata_indices::UPPERLEFT_Y)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(self.index)
    }

    fn scale_x(&self) -> f64 {
        self.metadata_struct
            .column(metadata_indices::SCALE_X)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(self.index)
    }

    fn scale_y(&self) -> f64 {
        self.metadata_struct
            .column(metadata_indices::SCALE_Y)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(self.index)
    }

    fn skew_x(&self) -> f64 {
        self.metadata_struct
            .column(metadata_indices::SKEW_X)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(self.index)
    }

    fn skew_y(&self) -> f64 {
        self.metadata_struct
            .column(metadata_indices::SKEW_Y)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(self.index)
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

    fn outdb_url(&self) -> Option<&str> {
        let url_array = self
            .metadata_struct
            .column(band_metadata_indices::OUTDB_URL)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Expected StringArray for outdb_url");

        if url_array.is_null(self.band_index) {
            None
        } else {
            Some(url_array.value(self.band_index))
        }
    }

    fn outdb_band_id(&self) -> Option<u32> {
        let band_id_array = self
            .metadata_struct
            .column(band_metadata_indices::OUTDB_BAND_ID)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("Expected UInt32Array for outdb_band_id");

        if band_id_array.is_null(self.band_index) {
            None
        } else {
            Some(band_id_array.value(self.band_index))
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
    crs: &'a StringArray,
    bbox: &'a StructArray,
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

        let crs = raster_struct
            .column(raster_indices::CRS)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        let bbox = raster_struct
            .column(raster_indices::BBOX)
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

        Self {
            metadata,
            crs,
            bbox,
            bands,
        }
    }

    /// Access the bounding box for this raster
    pub fn bounding_box(&self, raster_index: usize) -> Option<BoundingBox> {
        if self.bbox.is_null(raster_index) {
            None
        } else {
            Some(BoundingBox {
                min_x: self
                    .bbox
                    .column(bounding_box_indices::MIN_X)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .value(raster_index),
                min_y: self
                    .bbox
                    .column(bounding_box_indices::MIN_Y)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .value(raster_index),
                max_x: self
                    .bbox
                    .column(bounding_box_indices::MAX_X)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .value(raster_index),
                max_y: self
                    .bbox
                    .column(bounding_box_indices::MAX_Y)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .value(raster_index),
            })
        }
    }
}

impl<'a> RasterRef for RasterRefImpl<'a> {
    fn metadata(&self) -> &dyn MetadataRef {
        &self.metadata
    }

    fn crs(&self) -> Option<&str> {
        if self.crs.is_null(self.bands.raster_index) {
            None
        } else {
            Some(&self.crs.value(self.bands.raster_index))
        }
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
    /// URL for OutDb reference (only used when storage_type == OutDbRef)
    pub outdb_url: Option<String>,
    /// Band ID within the OutDb resource (only used when storage_type == OutDbRef)
    pub outdb_band_id: Option<u32>,
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
    pub const BBOX: &str = "bbox";
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
    pub const OUTDB_URL: &str = "outdb_url";
    pub const OUTDB_BAND_ID: &str = "outdb_band_id";
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
    pub const OUTDB_URL: usize = 3;
    pub const OUTDB_BAND_ID: usize = 4;
}

mod band_indices {
    pub const METADATA: usize = 0;
    pub const DATA: usize = 1;
}

mod raster_indices {
    pub const METADATA: usize = 0;
    pub const CRS: usize = 1;
    pub const BBOX: usize = 2;
    pub const BANDS: usize = 3;
}

#[cfg(test)]
mod tests {
    use crate::crs::lnglat;

    use super::*;

    #[test]
    fn sedona_type_arrow() {
        let sedona_type = SedonaType::Arrow(DataType::Int32);
        assert_eq!(sedona_type.storage_type(), &DataType::Int32);
        assert_eq!(sedona_type, SedonaType::Arrow(DataType::Int32));
        assert!(sedona_type.match_signature(&SedonaType::Arrow(DataType::Int32)));
        assert!(!sedona_type.match_signature(&SedonaType::Arrow(DataType::Utf8)));
    }

    #[test]
    fn sedona_type_wkb() {
        assert_eq!(WKB_GEOMETRY, WKB_GEOMETRY);
        assert_eq!(
            SedonaType::from_storage_field(&WKB_GEOMETRY.to_storage_field("", true).unwrap())
                .unwrap(),
            WKB_GEOMETRY
        );

        assert!(WKB_GEOMETRY.match_signature(&WKB_GEOMETRY));
    }

    #[test]
    fn sedona_type_wkb_view() {
        assert_eq!(WKB_VIEW_GEOMETRY.storage_type(), &DataType::BinaryView);
        assert_eq!(WKB_VIEW_GEOGRAPHY.storage_type(), &DataType::BinaryView);

        assert_eq!(WKB_VIEW_GEOMETRY, WKB_VIEW_GEOMETRY);
        assert_eq!(WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOGRAPHY);

        let storage_field = WKB_VIEW_GEOMETRY.to_storage_field("", true).unwrap();
        assert_eq!(
            SedonaType::from_storage_field(&storage_field).unwrap(),
            WKB_VIEW_GEOMETRY
        );
    }

    #[test]
    fn sedona_type_wkb_geography() {
        assert_eq!(WKB_GEOGRAPHY, WKB_GEOGRAPHY);
        assert_eq!(
            SedonaType::from_storage_field(&WKB_GEOGRAPHY.to_storage_field("", true).unwrap())
                .unwrap(),
            WKB_GEOGRAPHY
        );

        assert!(WKB_GEOGRAPHY.match_signature(&WKB_GEOGRAPHY));
        assert!(!WKB_GEOGRAPHY.match_signature(&WKB_GEOMETRY));
    }

    #[test]
    fn sedona_type_to_string() {
        assert_eq!(SedonaType::Arrow(DataType::Int32).to_string(), "Int32");
        assert_eq!(WKB_GEOMETRY.to_string(), "Wkb");
        assert_eq!(WKB_GEOGRAPHY.to_string(), "Wkb(Spherical)");
        assert_eq!(WKB_VIEW_GEOMETRY.to_string(), "WkbView");
        assert_eq!(WKB_VIEW_GEOGRAPHY.to_string(), "WkbView(Spherical)");
        assert_eq!(
            SedonaType::Wkb(Edges::Planar, lnglat()).to_string(),
            "Wkb(ogc:crs84)"
        );

        let projjson_value: Value = r#"{}"#.parse().unwrap();
        let projjson_crs = deserialize_crs(&projjson_value).unwrap();
        assert_eq!(
            SedonaType::Wkb(Edges::Planar, projjson_crs).to_string(),
            "Wkb({...})"
        );
    }

    #[test]
    fn sedona_logical_type_name() {
        assert_eq!(WKB_GEOMETRY.logical_type_name(), "geometry");
        assert_eq!(WKB_GEOGRAPHY.logical_type_name(), "geography");

        assert_eq!(
            SedonaType::Arrow(DataType::Int32).logical_type_name(),
            "int32"
        );

        assert_eq!(
            SedonaType::Arrow(DataType::Utf8).logical_type_name(),
            "utf8"
        );
        assert_eq!(
            SedonaType::Arrow(DataType::Utf8View).logical_type_name(),
            "utf8"
        );

        assert_eq!(
            SedonaType::Arrow(DataType::Binary).logical_type_name(),
            "binary"
        );
        assert_eq!(
            SedonaType::Arrow(DataType::BinaryView).logical_type_name(),
            "binary"
        );

        assert_eq!(
            SedonaType::Arrow(DataType::Duration(arrow_schema::TimeUnit::Microsecond))
                .logical_type_name(),
            "duration"
        );

        assert_eq!(
            SedonaType::Arrow(DataType::List(
                Field::new("item", DataType::Int32, true).into()
            ))
            .logical_type_name(),
            "list"
        );
        assert_eq!(
            SedonaType::Arrow(DataType::ListView(
                Field::new("item", DataType::Int32, true).into()
            ))
            .logical_type_name(),
            "list"
        );

        assert_eq!(
            SedonaType::Arrow(DataType::Dictionary(
                Box::new(DataType::Int32),
                Box::new(DataType::Binary)
            ))
            .logical_type_name(),
            "binary"
        );

        assert_eq!(
            SedonaType::Arrow(DataType::RunEndEncoded(
                Field::new("ends", DataType::Int32, true).into(),
                Field::new("values", DataType::Binary, true).into()
            ))
            .logical_type_name(),
            "binary"
        );
    }

    #[test]
    fn geoarrow_serialize() {
        assert_eq!(serialize_edges_and_crs(&Edges::Planar, &Crs::None), "{}");
        assert_eq!(
            serialize_edges_and_crs(&Edges::Planar, &lnglat()),
            r#"{"crs":"OGC:CRS84"}"#
        );
        assert_eq!(
            serialize_edges_and_crs(&Edges::Spherical, &Crs::None),
            r#"{"edges":"spherical"}"#
        );
        assert_eq!(
            serialize_edges_and_crs(&Edges::Spherical, &lnglat()),
            r#"{"edges":"spherical","crs":"OGC:CRS84"}"#
        );
    }

    #[test]
    fn geoarrow_serialize_roundtrip() -> Result<()> {
        // Check configuration resulting in empty metadata
        assert_eq!(
            deserialize_edges_and_crs(&Some(serialize_edges_and_crs(&Edges::Planar, &Crs::None)))?,
            (Edges::Planar, Crs::None)
        );

        // Check configuration with non-empty metadata for both edges and crs
        assert_eq!(
            deserialize_edges_and_crs(&Some(serialize_edges_and_crs(
                &Edges::Spherical,
                &lnglat()
            )))?,
            (Edges::Spherical, lnglat())
        );

        Ok(())
    }

    #[test]
    fn geoarrow_deserialize_invalid() {
        let bad_json =
            ExtensionType::new("geoarrow.wkb", DataType::Binary, Some(r#"{"#.to_string()));
        assert!(SedonaType::from_extension_type(bad_json)
            .unwrap_err()
            .message()
            .contains("Error deserializing GeoArrow metadata"));

        let bad_type =
            ExtensionType::new("geoarrow.wkb", DataType::Binary, Some(r#"[]"#.to_string()));
        assert!(SedonaType::from_extension_type(bad_type)
            .unwrap_err()
            .message()
            .contains("Expected GeoArrow metadata as JSON object"));

        let bad_edges_type = ExtensionType::new(
            "geoarrow.wkb",
            DataType::Binary,
            Some(r#"{"edges": []}"#.to_string()),
        );
        assert!(SedonaType::from_extension_type(bad_edges_type)
            .unwrap_err()
            .message()
            .contains("Unsupported edges JSON type"));

        let bad_edges_value = ExtensionType::new(
            "geoarrow.wkb",
            DataType::Binary,
            Some(r#"{"edges": "gazornenplat"}"#.to_string()),
        );
        assert!(SedonaType::from_extension_type(bad_edges_value)
            .unwrap_err()
            .message()
            .contains("Unsupported edges value"));
    }

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
        };

        let epsg4326 = "EPSG:4326";
        builder
            .start_raster(&metadata, Some(&epsg4326), metadata.bounding_box.as_ref())
            .unwrap();

        let band_metadata = BandMetadata {
            nodata_value: Some(vec![255u8]),
            storage_type: StorageType::InDb,
            datatype: BandDataType::UInt8,
            outdb_url: None,
            outdb_band_id: None,
        };

        // Add a single band with some test data using the correct API
        let test_data = vec![1u8; 100]; // 10x10 raster with value 1
        builder.band_data_writer().append_value(&test_data);
        builder.finish_band(band_metadata).unwrap();
        let result = builder.finish_raster();
        assert!(result.is_ok());

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

        let bbox = raster.bounding_box(0).unwrap();
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

        let crs = raster.crs().unwrap();
        assert_eq!(crs, epsg4326);

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
        };

        builder.start_raster(&metadata, None, None).unwrap();

        // Add three bands using the correct API
        for band_idx in 0..3 {
            let band_metadata = BandMetadata {
                nodata_value: Some(vec![255u8]),
                storage_type: StorageType::InDb,
                datatype: BandDataType::UInt8,
                outdb_url: None,
                outdb_band_id: None,
            };

            let test_data = vec![band_idx as u8; 25]; // 5x5 raster
            builder.band_data_writer().append_value(&test_data);
            builder.finish_band(band_metadata).unwrap();
        }

        let result = builder.finish_raster();
        assert!(result.is_ok());

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
        };

        source_builder
            .start_raster(
                &original_metadata,
                None,
                original_metadata.bounding_box.as_ref(),
            )
            .unwrap();

        let band_metadata = BandMetadata {
            nodata_value: Some(vec![255u8]),
            storage_type: StorageType::InDb,
            datatype: BandDataType::UInt8,
            outdb_url: None,
            outdb_band_id: None,
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
            .start_raster(
                source_raster.metadata(),
                source_raster.crs(),
                source_raster.bounding_box(0).as_ref(),
            )
            .unwrap();

        // Add new band data while preserving original metadata
        let new_band_metadata = BandMetadata {
            nodata_value: None,
            storage_type: StorageType::InDb,
            datatype: BandDataType::UInt16,
            outdb_url: None,
            outdb_band_id: None,
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

        let target_bbox = target_raster.bounding_box(0).unwrap();
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
            };

            builder.start_raster(&metadata, None, None).unwrap();

            let band_metadata = BandMetadata {
                nodata_value: Some(vec![255u8]),
                storage_type: StorageType::InDb,
                datatype: BandDataType::UInt8,
                outdb_url: None,
                outdb_band_id: None,
            };

            let size = (raster_idx + 1) * (raster_idx + 1);
            let test_data = vec![raster_idx as u8; size];
            builder.band_data_writer().append_value(&test_data);
            builder.finish_band(band_metadata).unwrap();
            let result = builder.finish_raster();
            assert!(result.is_ok());
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
        assert_eq!(raster_fields.len(), 4, "Expected exactly 4 raster fields");
        assert_eq!(
            raster_fields[raster_indices::METADATA].name(),
            column::METADATA,
            "Raster metadata index mismatch"
        );
        assert_eq!(
            raster_fields[raster_indices::CRS].name(),
            column::CRS,
            "Raster CRS index mismatch"
        );
        assert_eq!(
            raster_fields[raster_indices::BBOX].name(),
            column::BBOX,
            "Raster BBOX index mismatch"
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
                8,
                "Expected exactly 8 metadata fields"
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
                5,
                "Expected exactly 5 band metadata fields"
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
            assert_eq!(
                band_metadata_fields[band_metadata_indices::OUTDB_URL].name(),
                column::OUTDB_URL,
                "Band metadata outdb_url index mismatch"
            );
            assert_eq!(
                band_metadata_fields[band_metadata_indices::OUTDB_BAND_ID].name(),
                column::OUTDB_BAND_ID,
                "Band metadata outdb_band_id index mismatch"
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

    #[test]
    fn test_band_data_type_conversion() {
        // Create a test raster with bands of different data types
        let mut builder = RasterBuilder::new(10);

        let metadata = RasterMetadata {
            width: 2,
            height: 2,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
            bounding_box: None,
        };

        builder.start_raster(&metadata, None, None).unwrap();

        // Test all BandDataType variants
        let test_cases = vec![
            (BandDataType::UInt8, vec![1u8, 2u8, 3u8, 4u8]),
            (
                BandDataType::UInt16,
                vec![1u8, 0u8, 2u8, 0u8, 3u8, 0u8, 4u8, 0u8],
            ), // little-endian u16
            (
                BandDataType::Int16,
                vec![255u8, 255u8, 254u8, 255u8, 253u8, 255u8, 252u8, 255u8],
            ), // little-endian i16
            (
                BandDataType::UInt32,
                vec![
                    1u8, 0u8, 0u8, 0u8, 2u8, 0u8, 0u8, 0u8, 3u8, 0u8, 0u8, 0u8, 4u8, 0u8, 0u8, 0u8,
                ],
            ), // little-endian u32
            (
                BandDataType::Int32,
                vec![
                    255u8, 255u8, 255u8, 255u8, 254u8, 255u8, 255u8, 255u8, 253u8, 255u8, 255u8,
                    255u8, 252u8, 255u8, 255u8, 255u8,
                ],
            ), // little-endian i32
            (
                BandDataType::Float32,
                vec![
                    0u8, 0u8, 128u8, 63u8, 0u8, 0u8, 0u8, 64u8, 0u8, 0u8, 64u8, 64u8, 0u8, 0u8,
                    128u8, 64u8,
                ],
            ), // little-endian f32: 1.0, 2.0, 3.0, 4.0
            (
                BandDataType::Float64,
                vec![
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 240u8, 63u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    64u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 8u8, 64u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    16u8, 64u8,
                ],
            ), // little-endian f64: 1.0, 2.0, 3.0, 4.0
        ];

        for (expected_data_type, test_data) in test_cases {
            let band_metadata = BandMetadata {
                nodata_value: None,
                storage_type: StorageType::InDb,
                datatype: expected_data_type.clone(),
                outdb_url: None,
                outdb_band_id: None,
            };

            builder.band_data_writer().append_value(&test_data);
            builder.finish_band(band_metadata).unwrap();
        }

        builder.finish_raster().unwrap();
        let raster_array = builder.finish().unwrap();

        // Test the data type conversion for each band
        let iterator = raster_iterator(&raster_array);
        let raster = iterator.get(0).unwrap();
        let bands = raster.bands();

        assert_eq!(bands.len(), 7, "Expected 7 bands for all data types");

        // Verify each band returns the correct data type
        let expected_types = vec![
            BandDataType::UInt8,
            BandDataType::UInt16,
            BandDataType::Int16,
            BandDataType::UInt32,
            BandDataType::Int32,
            BandDataType::Float32,
            BandDataType::Float64,
        ];

        for (i, expected_type) in expected_types.iter().enumerate() {
            let band = bands.band(i).unwrap();
            let band_metadata = band.metadata();
            let actual_type = band_metadata.data_type();

            assert_eq!(
                actual_type, *expected_type,
                "Band {} expected data type {:?}, got {:?}",
                i, expected_type, actual_type
            );
        }
    }

    #[test]
    fn test_outdb_metadata_fields() {
        // Test creating raster with OutDb reference metadata
        let mut builder = RasterBuilder::new(10);

        let metadata = RasterMetadata {
            width: 1024,
            height: 1024,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
            bounding_box: None,
        };

        builder.start_raster(&metadata, None, None).unwrap();

        // Test InDb band (should have null OutDb fields)
        let indb_band_metadata = BandMetadata {
            nodata_value: Some(vec![255u8]),
            storage_type: StorageType::InDb,
            datatype: BandDataType::UInt8,
            outdb_url: None,
            outdb_band_id: None,
        };

        let test_data = vec![1u8; 100];
        builder.band_data_writer().append_value(&test_data);
        builder.finish_band(indb_band_metadata).unwrap();

        // Test OutDbRef band (should have OutDb fields populated)
        let outdb_band_metadata = BandMetadata {
            nodata_value: None,
            storage_type: StorageType::OutDbRef,
            datatype: BandDataType::Float32,
            outdb_url: Some("s3://mybucket/satellite_image.tif".to_string()),
            outdb_band_id: Some(2),
        };

        // For OutDbRef, data field could be empty or contain metadata/thumbnail
        builder.band_data_writer().append_value(&[]);
        builder.finish_band(outdb_band_metadata).unwrap();

        builder.finish_raster().unwrap();
        let raster_array = builder.finish().unwrap();

        // Verify the band metadata
        let iterator = raster_iterator(&raster_array);
        let raster = iterator.get(0).unwrap();
        let bands = raster.bands();

        assert_eq!(bands.len(), 2);

        // Test InDb band
        let indb_band = bands.band(0).unwrap();
        let indb_metadata = indb_band.metadata();
        assert_eq!(indb_metadata.storage_type(), StorageType::InDb);
        assert_eq!(indb_metadata.data_type(), BandDataType::UInt8);
        assert!(indb_metadata.outdb_url().is_none());
        assert!(indb_metadata.outdb_band_id().is_none());
        assert_eq!(indb_band.data().len(), 100);

        // Test OutDbRef band
        let outdb_band = bands.band(1).unwrap();
        let outdb_metadata = outdb_band.metadata();
        assert_eq!(outdb_metadata.storage_type(), StorageType::OutDbRef);
        assert_eq!(outdb_metadata.data_type(), BandDataType::Float32);
        assert_eq!(
            outdb_metadata.outdb_url().unwrap(),
            "s3://mybucket/satellite_image.tif"
        );
        assert_eq!(outdb_metadata.outdb_band_id().unwrap(), 2);
        assert_eq!(outdb_band.data().len(), 0); // Empty data for OutDbRef
    }
}
