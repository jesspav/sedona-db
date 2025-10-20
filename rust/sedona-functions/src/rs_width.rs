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

use crate::executor::WkbExecutor;
use arrow_array::builder::UInt64Builder;
use arrow_schema::DataType;
use datafusion_common::error::{DataFusionError, Result};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// RS_Width() scalar UDF implementation
///
/// Extract the width of the raster
pub fn rs_width_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_width",
        vec![Arc::new(RS_Width {})],
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
struct RS_Width {}

impl SedonaScalarKernel for RS_Width {
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
        let rasters = args[0].to_array();
        let mut builder = UInt64Builder::with_capacity(args[0].len());

        for raster in rasters.iter() {
            match raster {
                Some(raster) => {
                    builder.append_value(raster.metadata().width());
                }
                None => builder.append_null(),
            }
        }

        Ok(ColumnarValue::from(builder.finish()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{create_array, ArrayRef};
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_testing::{create::create_array, testers::ScalarUdfTester};

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = rs_width_udf().into();
        assert_eq!(udf.name(), "rs_width");
        assert!(udf.documentation().is_some());
    }

    #[rstest]
    fn udf_invoke() {
        let raster_array = create_array(
            &[gen_raster(10, 12), None, gen_raster(30, 15)],
            &WKB_GEOMETRY,
        );
        let expected: ArrayRef = create_array!(UInt64, [Some(10), None, Some(30),]);
        assert_eq!(
            &x_tester.invoke_array(wkb_array.clone()).unwrap(),
            &expected_x
        );
    }

    /// Generate a raster with the specified width, height, and value.
    /// This should be improved and moved into sedona-testing
    fn gen_raster(width: usize, height: usize) -> StructArray {
        let mut builder = Raster::builder();

        let metadata = RasterMetadata {
            width,
            height,
            ..Default::default()
        };

        let band_metadata = BandMetadata {
            nodata_value: Some(vec![255u8]),
            storage_type: StorageType::InDb,
            datatype: BandDataType::UInt8,
        };

        builder.start_raster(&metadata, None, None).unwrap();

        let size = width * height * 8;
        let test_data = vec![value as u8; size];
        builder.band_data_writer().append_value(&test_data);
        builder.finish_band(band_metadata).unwrap();
        builder.finish_raster();

        builder.finish().unwrap()
    }
}
