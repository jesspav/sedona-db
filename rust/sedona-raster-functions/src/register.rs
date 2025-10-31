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
use sedona_expr::function_set::FunctionSet;

/// Export the set of functions defined in this crate
pub fn default_function_set() -> FunctionSet {
    let mut function_set = FunctionSet::new();

    macro_rules! register_scalar_udfs {
        ($function_set:expr, $($udf:expr),* $(,)?) => {
            $(
                $function_set.insert_scalar_udf($udf());
            )*
        };
    }

    macro_rules! register_aggregate_udfs {
        ($function_set:expr, $($udf:expr),* $(,)?) => {
            $(
                $function_set.insert_aggregate_udf($udf());
            )*
        };
    }

    register_scalar_udfs!(function_set, crate::rs_size::rs_width_udf,);

    register_aggregate_udfs!(
        function_set,
        crate::st_analyze_aggr::st_analyze_aggr_udf,
        crate::st_collect::st_collect_udf,
        crate::st_envelope_aggr::st_envelope_aggr_udf,
        crate::st_intersection_aggr::st_intersection_aggr_udf,
        crate::st_union_aggr::st_union_aggr_udf,
    );

    function_set
}

/// Functions whose implementations are registered independently
///
/// These functions are included in the default function set; however,
/// it is useful to expose them individually for testing in crates that
/// implement them.
pub mod stubs {
    pub use crate::overlay::*;
    pub use crate::predicates::*;
    pub use crate::referencing::*;
    pub use crate::st_area::st_area_udf;
    pub use crate::st_azimuth::st_azimuth_udf;
    pub use crate::st_centroid::st_centroid_udf;
    pub use crate::st_length::st_length_udf;
    pub use crate::st_perimeter::st_perimeter_udf;
    pub use crate::st_setsrid::st_set_crs_with_engine_udf;
    pub use crate::st_setsrid::st_set_srid_with_engine_udf;
    pub use crate::st_transform::st_transform_udf;
}
