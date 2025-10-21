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
use std::sync::Arc;

use arrow_array::builder::Float64Builder;
use arrow_schema::{ArrowError, DataType};
use datafusion_common::{error::Result, scalar::ScalarValue};
use datafusion_expr::ColumnarValue;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_functions::executor::RasterExecutor;
use sedona_schema::datatypes::{BandMetadataRef, BandRef, RasterRef, SedonaType};
use sedona_raster::datatype_functions::{bytes_per_pixel, read_pixel_value};

/// RS_Value() implementation using [DistanceExt]
pub fn rs_value_impl() -> ScalarKernelRef {
    Arc::new(RSValue {})
}

#[derive(Debug)]
struct RSValue {}

impl SedonaScalarKernel for RSValue {
    fn return_type(&self, _arg_types: &[SedonaType]) -> Result<Option<SedonaType>, datafusion_common::DataFusionError> {
        Ok(Some(SedonaType::Arrow(DataType::Float64)))
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);

        // Extract coordinate and band arguments as scalars
        let x = match &args[1] {
            ColumnarValue::Scalar(scalar) => {
                let val = scalar.cast_to(&DataType::Int64).map_err(|e| datafusion_common::DataFusionError::Execution(format!("Failed to cast x coordinate: {}", e)))?;
                match val {
                    ScalarValue::Int64(Some(v)) => v as usize,
                    _ => return Err(datafusion_common::DataFusionError::NotImplemented("Invalid x coordinate".to_string())),
                }
            },
            _ => return Err(datafusion_common::DataFusionError::NotImplemented("Array x coordinates not supported".to_string())),
        };
        let y = match &args[2] {
            ColumnarValue::Scalar(scalar) => {
                let val = scalar.cast_to(&DataType::Int64).map_err(|e| datafusion_common::DataFusionError::Execution(format!("Failed to cast y coordinate: {}", e)))?;
                match val {
                    ScalarValue::Int64(Some(v)) => v as usize,
                    _ => return Err(datafusion_common::DataFusionError::NotImplemented("Invalid y coordinate".to_string())),
                }
            },
            _ => return Err(datafusion_common::DataFusionError::NotImplemented("Array y coordinates not supported".to_string())),
        };
        let band_index = match &args[3] {
            ColumnarValue::Scalar(scalar) => {
                let val = scalar.cast_to(&DataType::Int64).map_err(|e| datafusion_common::DataFusionError::Execution(format!("Failed to cast band index: {}", e)))?;
                match val {
                    ScalarValue::Int64(Some(v)) => (v as usize).saturating_sub(1),
                    _ => return Err(datafusion_common::DataFusionError::NotImplemented("Invalid band index".to_string())),
                }
            },
            _ => return Err(datafusion_common::DataFusionError::NotImplemented("Array band numbers not supported".to_string())),
        };

        let mut builder = Float64Builder::with_capacity(executor.num_iterations());

        executor.execute_raster_void(|_i, raster_opt| {
            match raster_opt {
                None => builder.append_null(),
                Some(raster) => {
                    match invoke_scalar(&raster, x, y, band_index) {
                        Ok(value) => builder.append_value(value),
                        Err(_) => builder.append_null(), // Handle errors by appending null
                    }
                }
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(raster: &dyn RasterRef, x: usize, y: usize, band_index: usize) -> Result<f64, ArrowError> {
    // Extract metadata from the raster
    let metadata = raster.metadata();
    let width = metadata.width() as usize;
    let height = metadata.height() as usize;
    
    // Check that x,y are within width/height
    if x >= width || y >= height {
        return Err(ArrowError::InvalidArgumentError(
            "Coordinates are outside raster bounds".to_string(),
        ));
    }
    
    // Get the band
    let bands = raster.bands();
    if band_index >= bands.len() {
        return Err(ArrowError::InvalidArgumentError(
            "Specified band does not exist".to_string(),
        ));
    }
    let band = bands.band(band_index).ok_or_else(|| ArrowError::InvalidArgumentError(
        "Failed to get band at index".to_string(),
    ))?; 
    let band_metadata = band.metadata();

    match band_metadata.storage_type() {
        sedona_schema::datatypes::StorageType::InDb => get_indb_pixel(band_metadata, &*band, x, y, width, height),
        sedona_schema::datatypes::StorageType::OutDbRef => get_outdb_pixel(band_metadata, x, y, width, height),
    }
}

fn get_indb_pixel(metadata: &dyn BandMetadataRef, band: &dyn BandRef, x: usize, y: usize, width: usize, _height: usize) -> Result<f64, ArrowError> {
    if let Some(_nodata_bytes) = metadata.nodata_value() {
        // TODO: Compare pixel value against nodata value
    }

    let data_type = metadata.data_type();
    let bytes_per_px = bytes_per_pixel(data_type.clone())?;
    let offset = (y * width + x) * bytes_per_px;
    
    let band_data = band.data();
    if offset + bytes_per_px > band_data.len() {
        return Err(ArrowError::InvalidArgumentError(
            "Pixel offset exceeds band data length".to_string(),
        ));
    }
    
    let pixel_bytes = &band_data[offset..offset + bytes_per_px];
    read_pixel_value(pixel_bytes, data_type)
}

fn get_outdb_pixel(metadata: &dyn BandMetadataRef, x: usize, y: usize, _width: usize, _height: usize) -> Result<f64, ArrowError> {
    use crate::dataset::get_outdb_dataset;
    
    let dataset = get_outdb_dataset(metadata)?;

    let band_index = match metadata.outdb_band_id() {
        Some(index) => index,
        None => {
            return Err(ArrowError::ParseError(
                "Raster band does not have a band index".to_string(),
            ))
        }
    };
    
    let band = dataset.rasterband(band_index as usize).map_err(|_| {
        ArrowError::ParseError("Failed to get raster band from dataset".to_string())
    })?;
    
    // Read a single pixel at the specified coordinates
    let pixel_data = band.read_as::<f64>((x as isize, y as isize), (1, 1), (1, 1), None)
        .map_err(|_| ArrowError::ParseError("Failed to read pixel data from GDAL".to_string()))?;
    
    Ok(pixel_data.data()[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, ArrayRef, Float64Array};
    use sedona_schema::datatypes::{BandDataType, BandMetadata, RasterBuilder, RasterMetadata, StorageType, RASTER};

    #[test]
    fn udf_invoke() {
        // Test with different band data types
        let band_types = vec![
            BandDataType::UInt8,
            BandDataType::Int16,
            BandDataType::UInt16,
            BandDataType::Int32,
            BandDataType::UInt32,
            BandDataType::Float32,
            BandDataType::Float64,
        ];

        for band_data_type in band_types {
            println!("Testing with band data type: {:?}", band_data_type);
            
            // Create test rasters with the current band data type
            let raster_array = create_indb_test_raster_array(band_data_type.clone());

            // Create the UDF and invoke it
            let kernel = RSValue {};
            /// Get pixel at (2,3) in band 1
            let args = vec![
                ColumnarValue::Array(raster_array),
                ColumnarValue::Scalar(ScalarValue::from(2i64)),
                ColumnarValue::Scalar(ScalarValue::from(3i64)),
                ColumnarValue::Scalar(ScalarValue::from(1i64))
            ];
            let arg_types = vec![
                RASTER,
                sedona_schema::datatypes::SedonaType::Arrow(DataType::Int64),
                sedona_schema::datatypes::SedonaType::Arrow(DataType::Int64),
                sedona_schema::datatypes::SedonaType::Arrow(DataType::Int64),
            ];

            let result = kernel.invoke_batch(&arg_types, &args).unwrap();

            // Check the result
            if let ColumnarValue::Array(result_array) = result {
                let pixel_array = result_array.as_any().downcast_ref::<Float64Array>().unwrap();

                assert_eq!(pixel_array.len(), 3);
                
                // Expected pixel value at (2,3) for 10x12 raster: row 3 * width 10 + col 2 = 32
                let expected_first = 32.0;
                assert_eq!(pixel_array.value(0), expected_first, "Failed for band type {:?}", band_data_type);
                assert!(pixel_array.is_null(1), "Second raster should be null for band type {:?}", band_data_type);
                
                // Expected pixel value at (2,3) for 30x15 raster: row 3 * width 30 + col 2 = 92  
                let expected_third = 92.0;
                assert_eq!(pixel_array.value(2), expected_third, "Failed for band type {:?}", band_data_type);
            } else {
                panic!("Expected array result for band type {:?}", band_data_type);
            }
        }
    }

    /// Create a test raster array with different widths for testing
    // TODO: Parameterize the creation of rasters and move the
    //       function to sedona-testing
    fn create_indb_test_raster_array(band_data_type: BandDataType) -> ArrayRef {
        let mut builder = RasterBuilder::new(3);

        // First raster: 10x12
        let metadata1 = RasterMetadata {
            width: 10,
            height: 12,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
            bounding_box: None,
        };

        let band_metadata = BandMetadata {
            nodata_value: Some(vec![255u8]),
            storage_type: StorageType::InDb,
            datatype: band_data_type.clone(),
            outdb_url: None,
            outdb_band_id: None,
        };

        builder.start_raster(&metadata1, None, None).unwrap();
        let test_data1 = gen_sequential(10 * 12, band_data_type.clone());
        builder.band_data_writer().append_value(&test_data1);
        builder.finish_band(band_metadata.clone()).unwrap();
        builder.finish_raster().unwrap();

        // Second raster: null
        builder.append_null().unwrap();

        // Third raster: 30x15
        let metadata3 = RasterMetadata {
            width: 30,
            height: 5,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
            bounding_box: None,
        };

        builder.start_raster(&metadata3, None, None).unwrap();
        let test_data3 = gen_sequential(30 * 15, band_data_type.clone());
        builder.band_data_writer().append_value(&test_data3);
        builder.finish_band(band_metadata).unwrap();
        builder.finish_raster().unwrap();

        Arc::new(builder.finish().unwrap())
    }

    /// Generates sequential pixel values of BandDataType for testing
    /// TODO: Add no-data values for testing
    fn gen_sequential(num_pixels: usize, band_data_type: BandDataType) -> Vec<u8> {
        let bytes_per_px = bytes_per_pixel(band_data_type.clone()).unwrap();
        let total_bytes = num_pixels * bytes_per_px;
        let mut data = Vec::with_capacity(total_bytes);

        for i in 0..num_pixels {
            match band_data_type {
                BandDataType::UInt8 => {
                    data.push(i  as u8);
                }
                BandDataType::Int16 => {
                    let bytes = (i as i16).to_le_bytes();
                    data.extend_from_slice(&bytes);
                }
                BandDataType::UInt16 => {
                    let bytes = (i as u16).to_le_bytes();
                    data.extend_from_slice(&bytes);
                }
                BandDataType::Int32 => {
                    let bytes = (i as i32).to_le_bytes();
                    data.extend_from_slice(&bytes);
                }
                BandDataType::UInt32 => {
                    let bytes = (i as u32).to_le_bytes();
                    data.extend_from_slice(&bytes);
                }
                BandDataType::Float32 => {
                    let bytes = (i as f32).to_le_bytes();
                    data.extend_from_slice(&bytes);
                }
                BandDataType::Float64 => {
                    let bytes = (i as f64).to_le_bytes();
                    data.extend_from_slice(&bytes);
                }
            }
        }

        data
    }

}
