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

/// RS_Value() scalar UDF implementation
///
/// Extract the value at a given point from the raster
/// TODO: Implement support for point geometry input
pub fn rs_value_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new_stub(
        "rs_value",
        ArgMatcher::new(
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_numeric(), 
                ArgMatcher::is_numeric(),
                ArgMatcher::is_numeric(),
            ],
            SedonaType::Arrow(DataType::Boolean),
        ),
        Volatility::Immutable,
        Some(rs_value_doc()),
    )
}
fn rs_value_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        format!(
            "Returns the value at the given point in the raster.",
        ),
        format!("RS_Value (raster: Raster, colX: Integer, colY: Integer, band: Integer)"),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_optional_argument("x", "Integer: X coordinate")
    .with_optional_argument("y", "Integer: Y coordinate")
    .with_optional_argument("point", "Point: Point geometry")
    .with_argument("band_id", "Integer: Band number (1-based index)")
    .with_sql_example(format!(
        "SELECT RS_Value(raster, x, y, band_id)",
    ))
    .with_sql_example(format!(
        "SELECT RS_Value(raster, point, band_id)",
    ))
    .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_expr::ScalarUDF;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = rs_value_udf().into();
        assert_eq!(udf.name(), "rs_value");
        assert!(udf.documentation().is_some());
    }
}