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
use arrow_schema::ArrowError;
use gdal::{Dataset, Metadata};
use sedona_schema::datatypes::{BandMetadataRef, StorageType};

/// Get the out-db dataset reference from a raster band.
pub fn outdb_dataset(metadata: &dyn BandMetadataRef) -> Result<Dataset, ArrowError> {
    if metadata.storage_type() != StorageType::OutDbRef {
        return Err(ArrowError::ParseError(
            "Raster band is not stored out-of-db".to_string(),
        ));
    }

    let url = match metadata.outdb_url() {
        Some(url) => url,
        None => {
            return Err(ArrowError::ParseError(
                "Raster band does not have an out-db URL".to_string(),
            ))
        }
    };

    // These datasets may appear in multiple rasters and called repeatedly.
    // Adding a caching layer here would improve performance.
    // Could also consider having a pool of these datasets - gdal dataset has a GetRefCount
    // that may be helpful for keeping track of references.
    open_outdb_band(&url)
}

fn open_outdb_band(url: &str) -> Result<Dataset, ArrowError> {
    let full_url = format!("/vsicurl/{}", url);
    let ds = Dataset::open(full_url).map_err(|e| ArrowError::ParseError(e.to_string()))?;
    Ok(ds)
}

/// Extract geotransform components from a GDAL dataset
/// Returns (upper_left_x, pixel_width, x_skew, upper_left_y, y_skew, pixel_height)
pub fn geotransform_components(
    dataset: &Dataset,
) -> Result<(f64, f64, f64, f64, f64, f64), ArrowError> {
    let geotransform = dataset
        .geo_transform()
        .map_err(|e| ArrowError::ParseError(format!("Failed to get geotransform: {e}")))?;
    Ok((
        geotransform[0], // Upper-left X coordinate
        geotransform[3], // Upper-left Y coordinate
        geotransform[1], // Pixel width (scale_x)
        geotransform[5], // Pixel height (scale_y, usually negative)
        geotransform[2], // X-direction skew
        geotransform[4], // Y-direction skew
    ))
}

/// Extract tile size from a GDAL dataset
/// If not provided, defaults to raster size
pub fn tile_size(dataset: &Dataset) -> Result<(usize, usize), ArrowError> {
    let raster_width = dataset.raster_size().0;
    let raster_height = dataset.raster_size().1;

    let tile_width = match dataset.metadata_item("TILEWIDTH", "") {
        Some(val) => val.parse::<usize>().unwrap_or(raster_width),
        None => raster_width,
    };
    let tile_height = match dataset.metadata_item("TILEHEIGHT", "") {
        Some(val) => val.parse::<usize>().unwrap_or(raster_height),
        None => raster_height,
    };

    Ok((tile_width, tile_height))
}

#[cfg(test)]
mod test {}
