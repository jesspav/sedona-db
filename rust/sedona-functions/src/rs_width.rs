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
use arrow_array::builder::UInt64Builder;
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::{
    datatypes::{RasterRef, SedonaType},
    matchers::ArgMatcher,
};

/// RS_Width() scalar UDF implementation
///
/// Extract the width of the raster
pub fn rs_width_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_width",
        vec![Arc::new(RsWidth {})],
        Volatility::Immutable,
        Some(rs_width_doc()),
    )
}

fn rs_width_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        format!("Return the width component of a raster",),
        format!("RS_Width(raster: Raster)"),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_sql_example(format!("SELECT RS_Width(raster)",))
    .build()
}

#[derive(Debug)]
struct RsWidth {}

impl SedonaScalarKernel for RsWidth {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster()],
            SedonaType::Arrow(DataType::UInt64),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let mut builder = UInt64Builder::with_capacity(executor.num_iterations());

        executor.execute_raster_void(|_i, raster_opt| {
            match raster_opt {
                None => builder.append_null(),
                Some(raster) => {
                    let width = raster.metadata().width();
                    builder.append_value(width);
                }
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, ArrayRef, UInt64Array};
    use datafusion_expr::ScalarUDF;
    use sedona_schema::datatypes::{
        BandDataType, BandMetadata, RasterBuilder, RasterMetadata, StorageType, RASTER,
    };

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = rs_width_udf().into();
        assert_eq!(udf.name(), "rs_width");
        assert!(udf.documentation().is_some());
    }

    #[test]
    fn udf_invoke() {
        // Create test rasters with different widths
        let raster_array = create_test_raster_array();

        // Create the UDF and invoke it
        let kernel = RsWidth {};
        let args = vec![ColumnarValue::Array(raster_array)];
        let arg_types = vec![RASTER];

        let result = kernel.invoke_batch(&arg_types, &args).unwrap();

        // Check the result
        if let ColumnarValue::Array(result_array) = result {
            let width_array = result_array.as_any().downcast_ref::<UInt64Array>().unwrap();

            assert_eq!(width_array.len(), 3);
            assert_eq!(width_array.value(0), 10); // First raster width
            assert!(width_array.is_null(1)); // Second raster is null
            assert_eq!(width_array.value(2), 30); // Third raster width
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
            datatype: BandDataType::UInt8,
            outdb_url: None,
            outdb_band_id: None,
        };

        builder.start_raster(&metadata1, None, None).unwrap();
        let test_data1 = vec![1u8; 10 * 12]; // width * height
        builder.band_data_writer().append_value(&test_data1);
        builder.finish_band(band_metadata.clone()).unwrap();
        builder.finish_raster().unwrap();

        // Second raster: null
        builder.append_null().unwrap();

        // Third raster: 30x15
        let metadata3 = RasterMetadata {
            width: 30,
            height: 15,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
            bounding_box: None,
        };

        builder.start_raster(&metadata3, None, None).unwrap();
        let test_data3 = vec![3u8; 30 * 15]; // width * height
        builder.band_data_writer().append_value(&test_data3);
        builder.finish_band(band_metadata).unwrap();
        builder.finish_raster().unwrap();

        Arc::new(builder.finish().unwrap())
    }
}
