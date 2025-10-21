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
use gdal::Dataset;
use sedona_schema::datatypes::{BandMetadataRef, StorageType};

/// Get the out-db dataset reference from a raster band.
pub fn get_outdb_dataset(metadata: &dyn BandMetadataRef) -> Result<Dataset, ArrowError> {
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
    open_outdb_band(&url)
}

fn open_outdb_band(url: &str) -> Result<Dataset, ArrowError> {
    let full_url = format!("/vsicurl/{}", url);
    let ds = Dataset::open(full_url).map_err(|e| ArrowError::ParseError(e.to_string()))?;
    Ok(ds)
}
