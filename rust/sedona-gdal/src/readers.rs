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

use crate::dataset::{geotransform_components, tile_size};
use arrow_array::StructArray;
use arrow_schema::ArrowError;
use gdal::raster::{GdalDataType, RasterBand};
use gdal::Dataset;
use sedona_raster::datatype_functions::{bytes_per_pixel, f64_to_bandtype_bytes};
use sedona_schema::datatypes::{
    BandDataType, BandMetadata, RasterBuilder, RasterMetadata, StorageType,
};
use std::sync::Arc;

pub fn read_raster(filepath: &str) -> Result<Arc<StructArray>, ArrowError> {
    let dataset = Dataset::open(filepath.to_string())
        .map_err(|err| ArrowError::ParseError(err.to_string()))?;

    // Extract geotransform components
    let (origin_x, origin_y, pixel_width, pixel_height, rotation_x, rotation_y) =
        geotransform_components(&dataset)?;

    let (raster_width, raster_height) = dataset.raster_size();

    let (tile_width, tile_height) = tile_size(&dataset)?;

    let x_tile_count = (raster_width + tile_width - 1) / tile_width;
    let y_tile_count = (raster_height + tile_height - 1) / tile_height;

    let mut raster_builder = RasterBuilder::new(x_tile_count * y_tile_count);
    let band_count = dataset.raster_count();

    for tile_y in 0..y_tile_count {
        for tile_x in 0..x_tile_count {
            let x_offset = tile_x * tile_width;
            let y_offset = tile_y * tile_height;

            // Calculate geographic coordinates for this tile
            // using the geotransform from the original raster
            let tile_origin_x =
                origin_x + (x_offset as f64) * pixel_width + (y_offset as f64) * rotation_x;
            let tile_origin_y =
                origin_y + (x_offset as f64) * rotation_y + (y_offset as f64) * pixel_height;

            // Create raster metadata for this tile with actual geotransform values
            let tile_metadata = RasterMetadata {
                width: tile_width as u64,
                height: tile_height as u64,
                upperleft_x: tile_origin_x,
                upperleft_y: tile_origin_y,
                scale_x: pixel_width,
                scale_y: pixel_height,
                skew_x: rotation_x,
                skew_y: rotation_y,
                bounding_box: None, // TODO: should we calculate bounding box here?
            };

            raster_builder.start_raster(&tile_metadata, None, None)?;

            for band_number in 1..=band_count {
                let band: RasterBand = dataset.rasterband(band_number).unwrap();
                // This should be the same as tile width/height, except for edge tiles
                // but we would need to update the width/height in the metadata above then.
                // For now, fail if sizes don't match.
                let (x_size, y_size) = band.size();
                if x_size != tile_width || y_size != tile_height {
                    return Err(ArrowError::ParseError(format!(
                        "Band size ({}, {}) does not match expected tile size ({}, {})",
                        x_size, y_size, tile_width, tile_height
                    )));
                }

                let data_type = gdaldatatype_to_banddatatype(band.band_type())?;
                let data_type_bytes = bytes_per_pixel(data_type.clone())?;
                let buffer_size_bytes = x_size * y_size * data_type_bytes.clone();

                // Get a mutable buffer slice for GDAL to write directly into
                let (buffer, slice) = raster_builder.get_band_buffer_slice(buffer_size_bytes);

                // TODO: Do we need resampling? If so set buffer_size to different from window_size
                //       and have a ResampleAlgorithm.
                band.read_into_slice(
                    (x_offset as isize, y_offset as isize), // window_origin
                    (x_size, y_size),                       // window_size
                    (x_size, y_size),                       // buffer_size (no resampling)
                    slice,                                  // buffer
                    None,                                   // resampling algorithms
                )
                .map_err(|e| {
                    ArrowError::ParseError(format!("Failed to read band {band_number} {e}"))
                })?;

                raster_builder.commit_band_buffer(buffer);

                let nodata_value = match band.no_data_value() {
                    Some(val) => Some(f64_to_bandtype_bytes(val, data_type.clone())?),
                    None => None,
                };

                let band_metadata = BandMetadata {
                    nodata_value: nodata_value,
                    storage_type: StorageType::InDb,
                    datatype: data_type,
                    outdb_url: None,
                    outdb_band_id: None,
                };

                // Finalize the band
                raster_builder.finish_band(band_metadata)?;
            }

            // Finalize the raster
            raster_builder.finish_raster()?;
        }
    }

    // Finalize the raster struct array
    let raster_struct = raster_builder.finish()?;
    Ok(Arc::new(raster_struct))
}

fn gdaldatatype_to_banddatatype(gdal_data_type: GdalDataType) -> Result<BandDataType, ArrowError> {
    match gdal_data_type {
        GdalDataType::UInt8 => Ok(BandDataType::UInt8),
        GdalDataType::UInt16 => Ok(BandDataType::UInt16),
        GdalDataType::Int16 => Ok(BandDataType::Int16),
        GdalDataType::UInt32 => Ok(BandDataType::UInt32),
        GdalDataType::Int32 => Ok(BandDataType::Int32),
        GdalDataType::Float32 => Ok(BandDataType::Float32),
        GdalDataType::Float64 => Ok(BandDataType::Float64),
        _ => Err(ArrowError::InvalidArgumentError(format!(
            "Unsupported GDAL data type: {:?}",
            gdal_data_type
        ))),
    }
}

#[cfg(test)]
mod tests {
    // use super::*;
    // use sedona_raster::display_functions::pretty_print_indb;
    // use sedona_schema::datatypes::raster_iterator;

    #[test]
    fn test_load_raster() {
        // TODO: Add proper tests here.
        // To load a raster and view contents
        // for prototyping fun:
        //
        // let filepath = "<your_path>/test1.tiff";
        // let result = read_raster(filepath);
        // assert!(result.is_ok());
        //
        // To view loaded raster:
        // let raster_array = result.unwrap();
        // for raster in raster_iterator(&raster_array) {
        //    println!("{}", pretty_print_indb(&raster, 1, 2).unwrap());
        // }
    }
}
