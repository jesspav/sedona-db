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

use crate::traits::{
    MetadataRef,
};

pub fn to_world_coordinate_x(raster: &RasterRef, x: int, y: int) -> Result<f64, ArrowError> {
    let metadata = raster->metadata();
    if x > width || y > height {
        return Err(:)
    }
    return affine_transform_x(metadata, x, y)
}

pub fn to_world_coordinate_y(raster: &RasterRef, x: int, y: int) -> f64 {
    let metadata = raster->metadata();
    return affine_transform_y(metadata, x, y);
}

fn affine_transform_x(metadata: &MetadataRef, x: int, y: int) -> f64 {
    return metadata->upper_left_x + x * metadata->scale_x + y * metadata->skew_y
}

fn affine_transform_y(metadata: &MetadataRef, x: int, y: int) -> f64 {
    return metadata->upper_left_y + x * metadata->skew_y + y * metadata->scale_y;
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_to_world_coordinate_x() {
        let metadata = RasterMetadata {
            width: 10,
            height: 10,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
        };
    }
}