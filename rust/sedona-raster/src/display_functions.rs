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

use crate::datatype_functions::{bytes_per_pixel, read_pixel_value};
use arrow::error::ArrowError;
use sedona_schema::datatypes::{RasterRef, RasterRefImpl, StorageType};

/// Pretty print a raster band to a string with specified precision
pub fn pretty_print_indb(
    raster: &RasterRefImpl,
    band_number: usize,
    precision: usize, // TODO: change this to an optional format string
) -> Result<String, ArrowError> {
    let band = raster.bands().band(band_number).unwrap();
    let metadata = raster.metadata();
    let height = metadata.height() as usize;
    let width = metadata.width() as usize;
    let mut result = String::new();

    let slice = band.data() as &[u8];
    let data_type = band.metadata().data_type();
    if band.metadata().storage_type() != StorageType::InDb {
        return Err(ArrowError::InvalidArgumentError(
            "Pretty print indb not supported for non-InDb storage".to_string(),
        ));
    }
    let bytes_per_pixel = bytes_per_pixel(data_type.clone()).unwrap_or(1);
    for row in 0..height {
        for col in 0..width {
            let start = (row * width + col) * bytes_per_pixel;
            let end = start + bytes_per_pixel;
            let pixel_bytes = &slice[start..end];

            match read_pixel_value(pixel_bytes, data_type.clone()) {
                Ok(value) => result.push_str(&format!("{:8.*} ", precision, value)),
                Err(_) => result.push_str(&format!("{:>8} ", "?")), // Well-spaced question mark
            }
        }
        result.push('\n');
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sedona_schema::datatypes::{
        BandDataType, BandMetadata, RasterBuilder, RasterMetadata, StorageType,
    };

    #[test]
    fn test_pretty_print() {
        let mut raster_builder = RasterBuilder::new(1);

        let metadata1 = RasterMetadata {
            width: 3,
            height: 2,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
            bounding_box: None,
        };

        let band_data_type = BandDataType::Float32;
        let band_metadata = BandMetadata {
            nodata_value: Some(vec![255u8]),
            storage_type: StorageType::InDb,
            datatype: band_data_type.clone(),
            outdb_url: None,
            outdb_band_id: None,
        };

        raster_builder.start_raster(&metadata1, None, None).unwrap();
        let pixel_values: Vec<f32> = vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6111];
        let test_data1: Vec<u8> = pixel_values
            .iter()
            .flat_map(|&val| val.to_le_bytes())
            .collect();
        raster_builder.band_data_writer().append_value(&test_data1);
        raster_builder.finish_band(band_metadata.clone()).unwrap();
        raster_builder.finish_raster().unwrap();

        let raster_struct = raster_builder.finish().unwrap();
        let raster = sedona_schema::datatypes::RasterRefImpl::new(&raster_struct, 0);

        let pretty = pretty_print_indb(&raster, 0, 2).unwrap();

        let expected = "    1.10     2.20     3.30 \n    4.40     5.50     6.61 \n";
        assert_eq!(pretty, expected);
    }
}
