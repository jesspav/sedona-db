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
use sedona_raster::traits::RasterRef;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// RS_Width() scalar UDF implementation
///
/// Extract the width of the raster
pub fn rs_width_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_width",
        vec![Arc::new(RsSize {
            size_type: SizeType::Width,
        })],
        Volatility::Immutable,
        Some(rs_width_doc()),
    )
}

/// RS_Height() scalar UDF documentation
///
/// Extract the height of the raster
pub fn rs_height_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_height",
        vec![Arc::new(RsSize {
            size_type: SizeType::Height,
        })],
        Volatility::Immutable,
        Some(rs_height_doc()),
    )
}

fn rs_width_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return the width component of a raster".to_string(),
        "RS_Width(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_sql_example("SELECT RS_Width(raster)".to_string())
    .build()
}

fn rs_height_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return the height component of a raster".to_string(),
        "RS_Height(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_sql_example("SELECT RS_Height(raster)".to_string())
    .build()
}

#[derive(Debug, Clone)]
enum SizeType {
    Width,
    Height,
}

#[derive(Debug)]
struct RsSize {
    size_type: SizeType,
}

impl SedonaScalarKernel for RsSize {
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
                Some(raster) => match self.size_type {
                    SizeType::Width => {
                        let width = raster.metadata().width();
                        builder.append_value(width);
                    }
                    SizeType::Height => {
                        let height = raster.metadata().height();
                        builder.append_value(height);
                    }
                },
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, UInt64Array};
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::rasters::generate_test_rasters;

    #[test]
    fn udf_size() {
        let udf: ScalarUDF = rs_width_udf().into();
        assert_eq!(udf.name(), "rs_width");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = rs_height_udf().into();
        assert_eq!(udf.name(), "rs_height");
        assert!(udf.documentation().is_some());
    }

    #[rstest]
    fn udf_invoke(#[values(SizeType::Width, SizeType::Height)] st: SizeType) {
        let kernel = RsSize {
            size_type: st.clone(),
        };
        // 3 rasters, second one is null
        let rasters = generate_test_rasters(3, Some(1)).unwrap();

        // Create the UDF and invoke it
        let args = [ColumnarValue::Array(Arc::new(rasters))];
        let arg_types = vec![RASTER];

        let result = kernel.invoke_batch(&arg_types, &args).unwrap();

        // Check the result
        if let ColumnarValue::Array(result_array) = result {
            let size_array = result_array.as_any().downcast_ref::<UInt64Array>().unwrap();

            assert_eq!(size_array.len(), 3);

            match st.clone() {
                SizeType::Width => assert_eq!(size_array.value(0), 1), // First raster width
                SizeType::Height => assert_eq!(size_array.value(0), 2), // First raster height
            }
            assert!(size_array.is_null(1)); // Second raster is null
            match st.clone() {
                SizeType::Width => assert_eq!(size_array.value(2), 3), // Third raster width
                SizeType::Height => assert_eq!(size_array.value(2), 4), // Third raster height
            }
        } else {
            panic!("Expected array result");
        }
    }
}
