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
use std::{sync::Arc, vec};

use crate::executor::RasterExecutor;
use arrow_array::builder::StringBuilder;
use arrow_schema::ArrowError;
use arrow_schema::DataType;
use datafusion_common::{error::Result, scalar::ScalarValue};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::display_functions::write_band_to_builder;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// RS_AsMatrix() scalar UDF implementation
///
/// Extract the width of the raster
pub fn rs_asmatrix_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_asmatrix",
        vec![Arc::new(RsAsMatrix {})],
        Volatility::Immutable,
        Some(rs_asmatrix_doc()),
    )
}

fn rs_asmatrix_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        format!("Returns a string, that when printed, outputs the raster band as a pretty printed 2D matrix."),
        format!("RS_AsMatrix(raster: Raster, band_number: Numeric, postDecimalPrecision: Numeric)"),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_argument("band_number", "Numeric: Band number (1-based).")
    .with_argument("postDecimalPrecision", "Numeric: Number of digits after decimal point. Optional, default is 6.")
    .with_sql_example("SELECT RS_AsMatrix(raster, band_number)")
    .with_sql_example("SELECT RS_AsMatrix(raster, band_number, postDecimalPrecision)")
    .build()
}

#[derive(Debug)]
struct RsAsMatrix {}

impl SedonaScalarKernel for RsAsMatrix {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_numeric(),
                ArgMatcher::optional(ArgMatcher::is_numeric()),
            ],
            SedonaType::Arrow(DataType::Utf8),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let mut builder = StringBuilder::with_capacity(
            executor.num_iterations(),
            256 * executor.num_iterations(),
        );

        let band_number = extract_numeric_scalar(&args[1])? as usize;
        let precision = if args.len() > 2 {
            extract_numeric_scalar(&args[2])? as usize
        } else {
            6usize
        };

        executor.execute_raster_void(|_i, raster_opt| {
            match raster_opt {
                Some(raster) => {
                    let mut output = String::new(); // TODO: Perf can be improved here
                    match write_band_to_builder(&raster, band_number, precision, &mut output) {
                        Ok(_) => builder.append_value(&output),
                        Err(_) => builder.append_null(),
                    }
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn extract_numeric_scalar(arg: &ColumnarValue) -> Result<usize, ArrowError> {
    if let ColumnarValue::Scalar(scalar) = arg {
        match scalar {
            ScalarValue::Int8(Some(val)) => Ok(*val as usize),
            ScalarValue::Int16(Some(val)) => Ok(*val as usize),
            ScalarValue::Int32(Some(val)) => Ok(*val as usize),
            ScalarValue::Int64(Some(val)) => Ok(*val as usize),
            ScalarValue::UInt8(Some(val)) => Ok(*val as usize),
            ScalarValue::UInt16(Some(val)) => Ok(*val as usize),
            ScalarValue::UInt32(Some(val)) => Ok(*val as usize),
            ScalarValue::UInt64(Some(val)) => Ok(*val as usize),
            _ => Err(ArrowError::ParseError(
                "Failed to extract numeric scalar: unsupported type or null value".to_string(),
            )),
        }
    } else {
        Err(ArrowError::ParseError(
            "Failed to extract scalar value: expected scalar, got array".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, ArrayRef, StringArray};
    use datafusion_expr::ScalarUDF;
    use sedona_schema::datatypes::{
        BandDataType, BandMetadata, RasterBuilder, RasterMetadata, StorageType, RASTER,
    };

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = rs_asmatrix_udf().into();
        assert_eq!(udf.name(), "rs_asmatrix");
        assert!(udf.documentation().is_some());
    }

    #[test]
    fn udf_invoke() {
        let raster_array = create_test_raster_array();

        let kernel = RsAsMatrix {};
        let args = vec![
            ColumnarValue::Array(raster_array),
            ColumnarValue::Scalar(ScalarValue::from(1u64)),
        ];
        let arg_types = vec![
            RASTER,
            SedonaType::Arrow(DataType::UInt64),
            SedonaType::Arrow(DataType::Utf8),
        ];

        let result = kernel.invoke_batch(&arg_types, &args).unwrap();

        let expected_first = "1.000000 1.000000 1.000000 \n1.000000 1.000000 1.000000 \n";
        let expected_third = "3.000000 3.000000 \n3.000000 3.000000 \n3.000000 3.000000 \n";

        if let ColumnarValue::Array(result_array) = result {
            let string_array = result_array.as_any().downcast_ref::<StringArray>().unwrap();

            assert_eq!(string_array.len(), 3);
            assert_eq!(string_array.value(0), expected_first);
            assert!(string_array.is_null(1)); // Second raster is null
            assert_eq!(string_array.value(2), expected_third);
        } else {
            panic!("Expected array result");
        }
    }

    /// Create a test raster array with different widths for testing
    // TODO: Parameterize the creation of rasters and move the
    //       function to sedona-testing
    fn create_test_raster_array() -> ArrayRef {
        let mut builder = RasterBuilder::new(3);

        // First raster: 10x12
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

        let band_metadata = BandMetadata {
            nodata_value: Some(vec![255u8]),
            storage_type: StorageType::InDb,
            datatype: BandDataType::UInt8,
            outdb_url: None,
            outdb_band_id: None,
        };

        builder.start_raster(&metadata1, None, None).unwrap();
        let test_data1 = vec![1u8; 3 * 2]; // width * height
        builder.band_data_writer().append_value(&test_data1);
        builder.finish_band(band_metadata.clone()).unwrap();
        builder.finish_raster().unwrap();

        // Second raster: null
        builder.append_null().unwrap();

        // Third raster: 30x15
        let metadata3 = RasterMetadata {
            width: 2,
            height: 3,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
            bounding_box: None,
        };

        builder.start_raster(&metadata3, None, None).unwrap();
        let test_data3 = vec![3u8; 2 * 3]; // width * height
        builder.band_data_writer().append_value(&test_data3);
        builder.finish_band(band_metadata).unwrap();
        builder.finish_raster().unwrap();

        Arc::new(builder.finish().unwrap())
    }
}
