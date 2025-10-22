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

use arrow_array::StructArray;
use arrow_schema::ArrowError;
use gdal::raster::{GdalDataType, RasterBand, ResampleAlg};
use gdal::Dataset;
use sedona_raster::datatype_functions::{
    bytes_per_pixel, f64_to_bandtype_bytes, 
    cast_slice_to_u8, cast_slice_to_u16, cast_slice_to_i16, 
    cast_slice_to_u32, cast_slice_to_i32, cast_slice_to_f32, cast_slice_to_f64
};
use sedona_schema::datatypes::{
    BandDataType, BandMetadata, RasterBuilder, RasterMetadata, StorageType,
};
use std::sync::Arc;

fn load_raster(
    filepath: &str,
    band_indexes: &Vec<usize>, // 1-based index (GDAL convention)
    tile_width: Option<usize>, // Optional override, will use GDAL's block size if None
    tile_height: Option<usize>, // Optional override, will use GDAL's block size if None
) -> Result<Arc<StructArray>, ArrowError> {

    let dataset = Dataset::open(filepath.to_string())
        .map_err(|err| ArrowError::ParseError(err.to_string()))?;
    
    // Determine tile size - use GDAL's natural block size if not specified
    let (tile_width, tile_height) = if tile_width.is_some() && tile_height.is_some() {
        (tile_width.unwrap(), tile_height.unwrap())
    } else {
        // Get the natural block size from the first band
        let first_band = dataset.rasterband(band_indexes[0]).map_err(|e| {
            ArrowError::InvalidArgumentError(format!("Failed to read band {}: {e}", band_indexes[0]))
        })?;
        
        let (block_width, block_height) = first_band.block_size();
        (block_width, block_height)
    };
    
    println!("Using tile size: {}×{}", tile_width, tile_height);
    
    // Get the geotransform which contains scale, skew, and origin information
    let geotransform = dataset.geo_transform()
        .map_err(|e| ArrowError::ParseError(format!("Failed to get geotransform: {e}")))?;
    
    // Extract geotransform components
    // geotransform = [upperleft_x, scale_x, skew_x, upperleft_y, skew_y, scale_y]
    let (origin_x, pixel_width, rotation_x, origin_y, rotation_y, pixel_height) = (
        geotransform[0], // Upper-left X coordinate
        geotransform[1], // Pixel width (scale_x)
        geotransform[2], // X-direction skew
        geotransform[3], // Upper-left Y coordinate  
        geotransform[4], // Y-direction skew
        geotransform[5], // Pixel height (scale_y, usually negative)
    );


    let (raster_width, raster_height) = dataset.raster_size();
    let x_count = (raster_width + tile_width - 1) / tile_width;
    let y_count = (raster_height + tile_height - 1) / tile_height;

    let mut raster_builder = RasterBuilder::new(x_count * y_count);

    // FIXME: deal with the edge tiles when things don't divide evenly
    for tile_y in 0..y_count {
        for tile_x in 0..x_count {
            let x_offset = tile_x * tile_width;
            let y_offset = tile_y * tile_height;
            
            // Calculate the actual geographic coordinates for this tile
            // using the geotransform from the original raster
            let tile_origin_x = origin_x + (x_offset as f64) * pixel_width + (y_offset as f64) * rotation_x;
            let tile_origin_y = origin_y + (x_offset as f64) * rotation_y + (y_offset as f64) * pixel_height;
            
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
                bounding_box: None,
            };
            
            // Start the raster
            raster_builder.start_raster(&tile_metadata, None, None)?;

            for band_index in band_indexes {
                let band: RasterBand = dataset.rasterband(*band_index).map_err(|e| {
                    ArrowError::InvalidArgumentError(format!("Failed to read file: {filepath} {e}"))
                })?;

                let x_size = if x_offset + tile_width > raster_width {
                    raster_width - x_offset
                } else {
                    tile_width
                };
                let y_size = if y_offset + tile_height > raster_height {
                    raster_height - y_offset
                } else {
                    tile_height
                };

                let data_type = gdaldatatype_to_banddatatype(band.band_type())?;
                let data_type_bytes = bytes_per_pixel(data_type.clone())?;
                let buffer_size_bytes = x_size * y_size * data_type_bytes.clone();

                // Get a mutable buffer slice for GDAL to write directly into
                let (buffer, slice) = raster_builder.get_band_buffer_slice(buffer_size_bytes);

                // Note: If you want resampling, buffer_size can be different from window_size
                band.read_into_slice(
                    (x_offset as isize, y_offset as isize),   // window_origin
                    (x_size, y_size),                         // window_size
                    (x_size, y_size),                         // buffer_size (no resampling)
                    slice,                                    // buffer
                    Some(ResampleAlg::Average),               // TODO: consider other algorithms
                )
                    .map_err(|e| ArrowError::ParseError(format!("Failed to read file: {filepath} {e}")))?;
                
                // Commit the buffer to the band data
                raster_builder.commit_band_buffer(buffer);
                
                let nodata_value = match band.no_data_value() {
                    Some(val) => Some(f64_to_bandtype_bytes(val, data_type.clone())?),
                    None => None,
                };
                
                // Create band metadata
                let band_metadata = BandMetadata {
                    nodata_value: nodata_value,
                    storage_type: StorageType::InDb,
                    datatype: data_type, // Updated to use the correct variable
                    outdb_url: None,
                    outdb_band_id: None,
                };
                
                // Finish the band
                raster_builder.finish_band(band_metadata)?;
            }
            
            // Complete the raster
            raster_builder.finish_raster()?;
        }
    }

    let raster_struct = raster_builder.finish()?;
    Ok(Arc::new(raster_struct))
}

fn gdaldatatype_to_banddatatype(
    gdal_data_type: GdalDataType,
) -> Result<BandDataType, ArrowError> {
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
    use super::*;
    use gdal::Metadata;

    #[test]
    fn test_load_raster() {
        let filename = "/Users/jess/code/data/rasters/test1.tiff";
        let bands = vec![1];

        // Test with auto-detected tile size
        let _rasters = load_raster(filename, &bands, None, None).unwrap();
        
        // Test with custom tile size
        let _rasters = load_raster(filename, &bands, Some(128), Some(128)).unwrap();
    }

    #[test]
    fn test_gdal_tile_properties() {
        let filename = "/Users/jess/code/data/rasters/test1.tiff";
        
        let dataset = Dataset::open(filename).unwrap();
        
        println!("Dataset properties:");
        println!("- Raster size: {:?}", dataset.raster_size());
        println!("- Raster count: {}", dataset.raster_count());
        println!("- Driver: {:?}", dataset.driver().short_name());
        
        // Check for tile/block size on the first band
        if let Ok(band) = dataset.rasterband(1) {
            println!("\nBand 1 properties:");
            println!("- Band type: {:?}", band.band_type());
            println!("- Size: {:?}", band.size());
            
            // Check for block size (this is the natural tile/block size for the format)
            println!("- Block size: {:?}", band.block_size());
            
            // Check for overviews (pyramid levels)
            println!("- Overview count: {:?}", band.overview_count());
            
            // Check for no data value
            if let Some(nodata) = band.no_data_value() {
                println!("- No data value: {}", nodata);
            }
        }
        
        // Check dataset metadata for tiling information
        println!("\nDataset metadata:");
        for entry in dataset.metadata() {
            println!("- {}: {}", entry.key, entry.value);
        }
        
        // Check for specific metadata items related to tiling
        if let Some(tile_width) = dataset.metadata_item("TILEWIDTH", "") {
            println!("- TILEWIDTH: {}", tile_width);
        }
        if let Some(tile_height) = dataset.metadata_item("TILEHEIGHT", "") {
            println!("- TILEHEIGHT: {}", tile_height);
        }
        if let Some(block_x) = dataset.metadata_item("BLOCK_X_SIZE", "") {
            println!("- BLOCK_X_SIZE: {}", block_x);
        }
        if let Some(block_y) = dataset.metadata_item("BLOCK_Y_SIZE", "") {
            println!("- BLOCK_Y_SIZE: {}", block_y);
        }
    }
}