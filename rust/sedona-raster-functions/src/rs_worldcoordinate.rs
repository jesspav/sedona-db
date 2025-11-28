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
use arrow_array::builder::Float64Builder;
use arrow_schema::DataType;
use datafusion_common::{error::Result, exec_err, ScalarValue};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::affine_transformation::to_world_coordinate;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// RS_RasterToWorldCoordY() scalar UDF implementation
///
/// Extract the rastertoworldcoordy of the raster
pub fn rs_rastertoworldcoordy_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_rastertoworldcoordy",
        vec![Arc::new(RsCoordinateMapper { coord: Coord::Y })],
        Volatility::Immutable,
        Some(rs_rastertoworldcoordy_doc()),
    )
}

/// RS_RasterToWorldCoordX() scalar UDF documentation
///
/// Extract the rastertoworldcoordx of the raster
pub fn rs_rastertoworldcoordx_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_rastertoworldcoordx",
        vec![Arc::new(RsCoordinateMapper { coord: Coord::X })],
        Volatility::Immutable,
        Some(rs_rastertoworldcoordx_doc()),
    )
}

fn rs_rastertoworldcoordy_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the upper left Y coordinate of the given row and column of the given raster geometric units of the geo-referenced raster. If any out of bounds values are given, the Y coordinate of the assumed point considering existing raster pixel size and skew values will be returned.".to_string(),
        "RS_RasterToWorldCoordY(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_argument("x", "Integer: Column x into the raster")
    .with_argument("y", "Integer: Row y into the raster")
    .with_sql_example("SELECT RS_RasterToWorldCoordY(RS_Example(), 1, 1)".to_string())
    .build()
}

fn rs_rastertoworldcoordx_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the upper left X coordinate of the given row and column of the given raster geometric units of the geo-referenced raster. If any out of bounds values are given, the X coordinate of the assumed point considering existing raster pixel size and skew values will be returned.".to_string(),
        "RS_RasterToWorldCoordX(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_argument("x", "Integer: Column x into the raster")
    .with_argument("y", "Integer: Row y into the raster")
    .with_sql_example("SELECT RS_RasterToWorldCoordX(RS_Example(), 1, 1)".to_string())
    .build()
}

#[derive(Debug, Clone)]
enum Coord {
    X,
    Y,
}

#[derive(Debug)]
struct RsCoordinateMapper {
    coord: Coord,
}

impl SedonaScalarKernel for RsCoordinateMapper {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_integer(),
            ],
            SedonaType::Arrow(DataType::Float64),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let mut builder = Float64Builder::with_capacity(executor.num_iterations());

        let (x_opt, y_opt) = get_scalar_coord(&args[1], &args[2])?;
        match (x_opt, y_opt) {
            (None, _) | (_, None) => {
                for _ in 0..executor.num_iterations() {
                    builder.append_null();
                }
                return Ok(ColumnarValue::Array(Arc::new(builder.finish())));
            }
            _ => {}
        }

        let x = x_opt.unwrap();
        let y = y_opt.unwrap();

        executor.execute_raster_void(|_i, raster_opt| {
            match raster_opt {
                None  => builder.append_null(),
                Some(raster) => {
                  let (world_x, world_y) = to_world_coordinate(&raster, x, y)?;
                    match self.coord {
                        Coord::X => builder.append_value(world_x),
                        Coord::Y => builder.append_value(world_y),
                    };
                },
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn extract_positive_scalar(arg: &ColumnarValue) -> Result<Option<u64>> {
    match arg {
        ColumnarValue::Scalar(scalar) => {
            let i64_val = scalar.cast_to(&DataType::Int64)?;
            match i64_val {
                ScalarValue::Int64(Some(v)) if v >= 0 => Ok(Some(v as u64)),
                ScalarValue::Int64(None) => Ok(None),
                _ => exec_err!("Coordinate must be non-negative"),
            }
        }
        _ => exec_err!("Unsupported coordinate type, only integer scalar values are supported"),
    }
}

fn get_scalar_coord(
    x_arg: &ColumnarValue,
    y_arg: &ColumnarValue,
) -> Result<(Option<u64>, Option<u64>)> {
    let x_opt = extract_positive_scalar(x_arg)?;
    let y_opt = extract_positive_scalar(y_arg)?;
    Ok((x_opt, y_opt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn udf_docs() {
        let udf: ScalarUDF = rs_rastertoworldcoordy_udf().into();
        assert_eq!(udf.name(), "rs_rastertoworldcoordy");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = rs_rastertoworldcoordx_udf().into();
        assert_eq!(udf.name(), "rs_rastertoworldcoordx");
        assert!(udf.documentation().is_some());
    }

    #[rstest]
    fn udf_invoke(#[values(Coord::Y, Coord::X)] coord: Coord) {
        let udf = match coord {
            Coord::X => rs_rastertoworldcoordx_udf(),
            Coord::Y => rs_rastertoworldcoordy_udf(),
        };
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Arrow(DataType::Int32),
            ],
        );

        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        // At 0,0 expect the upper left corner of the test values
        let expected_values = match coord {
            Coord::X => vec![Some(1.0), None, Some(3.0)],
            Coord::Y => vec![Some(2.0), None, Some(4.0)],
        };
        let expected: Arc<dyn arrow_array::Array> =
            Arc::new(arrow_array::Float64Array::from(expected_values));

        let result = tester
            .invoke_array_scalar_scalar(Arc::new(rasters), 0_i32, 0_i32)
            .unwrap();
        assert_array_equal(&result, &expected);
    }
}
