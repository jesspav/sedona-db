use arrow_schema::ArrowError;
use sedona_schema::datatypes::BandDataType;

pub fn bytes_per_pixel(data_type: BandDataType) -> Result<usize, ArrowError> {
    match data_type {
        BandDataType::UInt8 => Ok(1),
        BandDataType::Int16 => Ok(2),
        BandDataType::UInt16 => Ok(2),
        BandDataType::Int32 => Ok(4),
        BandDataType::UInt32 => Ok(4),
        BandDataType::Float32 => Ok(4),
        BandDataType::Float64 => Ok(8),
    }
}

/// Extract a pixel value from raw bytes and convert to f64
pub fn read_pixel_value(bytes: &[u8], data_type: BandDataType) -> Result<f64, ArrowError> {
    let expected_bytes = bytes_per_pixel(data_type.clone())?;
    if bytes.len() != expected_bytes {
        return Err(ArrowError::InvalidArgumentError(
            "Invalid byte length for specified data type".to_string(),
        ));
    }

    match data_type {
        BandDataType::UInt8 => Ok(bytes[0] as f64),
        BandDataType::Int16 => {
            let value = i16::from_le_bytes([bytes[0], bytes[1]]);
            Ok(value as f64)
        }
        BandDataType::UInt16 => {
            let value = u16::from_le_bytes([bytes[0], bytes[1]]);
            Ok(value as f64)
        }
        BandDataType::Int32 => {
            let value = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            Ok(value as f64)
        }
        BandDataType::UInt32 => {
            let value = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            Ok(value as f64)
        }
        BandDataType::Float32 => {
            let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            Ok(value as f64)
        }
        BandDataType::Float64 => {
            let value = f64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]);
            Ok(value)
        }
    }
}

// Helper functions to cast byte slices to specific types
pub fn cast_slice_to_u8(slice: &mut [u8]) -> &mut [u8] {
    slice
}

pub fn cast_slice_to_u16(slice: &mut [u8]) -> &mut [u16] {
    let len = slice.len() / 2;
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u16, len) }
}

pub fn cast_slice_to_i16(slice: &mut [u8]) -> &mut [i16] {
    let len = slice.len() / 2;
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut i16, len) }
}

pub fn cast_slice_to_u32(slice: &mut [u8]) -> &mut [u32] {
    let len = slice.len() / 4;
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u32, len) }
}

pub fn cast_slice_to_i32(slice: &mut [u8]) -> &mut [i32] {
    let len = slice.len() / 4;
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut i32, len) }
}

pub fn cast_slice_to_f32(slice: &mut [u8]) -> &mut [f32] {
    let len = slice.len() / 4;
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut f32, len) }
}

pub fn cast_slice_to_f64(slice: &mut [u8]) -> &mut [f64] {
    let len = slice.len() / 8;
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut f64, len) }
}

pub fn f64_to_bandtype_bytes(value: f64, data_type: BandDataType) -> Result<Vec<u8>, ArrowError> {
    match data_type {
        BandDataType::UInt8 => Ok(vec![value as u8]),
        BandDataType::Int16 => Ok((value as i16).to_le_bytes().to_vec()),
        BandDataType::UInt16 => Ok((value as u16).to_le_bytes().to_vec()),
        BandDataType::Int32 => Ok((value as i32).to_le_bytes().to_vec()),
        BandDataType::UInt32 => Ok((value as u32).to_le_bytes().to_vec()),
        BandDataType::Float32 => Ok((value as f32).to_le_bytes().to_vec()),
        BandDataType::Float64 => Ok((value as f64).to_le_bytes().to_vec()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_bytes_per_pixel_tests() {
        assert_eq!(bytes_per_pixel(BandDataType::UInt8).unwrap(), 1);
        assert_eq!(bytes_per_pixel(BandDataType::Int16).unwrap(), 2);
        assert_eq!(bytes_per_pixel(BandDataType::UInt16).unwrap(), 2);
        assert_eq!(bytes_per_pixel(BandDataType::Int32).unwrap(), 4);
        assert_eq!(bytes_per_pixel(BandDataType::UInt32).unwrap(), 4);
        assert_eq!(bytes_per_pixel(BandDataType::Float32).unwrap(), 4);
        assert_eq!(bytes_per_pixel(BandDataType::Float64).unwrap(), 8);
    }

    #[test]
    fn basic_read_pixel_value_tests() {
        let nodataval = 17.0 as f64;
        assert_eq!(
            read_pixel_value(&[17u8], BandDataType::UInt8).unwrap(),
            nodataval
        );
        assert_eq!(
            read_pixel_value(&17i16.to_le_bytes(), BandDataType::Int16).unwrap(),
            nodataval
        );
        assert_eq!(
            read_pixel_value(&17u16.to_le_bytes(), BandDataType::UInt16).unwrap(),
            nodataval
        );
        assert_eq!(
            read_pixel_value(&17i32.to_le_bytes(), BandDataType::Int32).unwrap(),
            nodataval
        );
        assert_eq!(
            read_pixel_value(&17u32.to_le_bytes(), BandDataType::UInt32).unwrap(),
            nodataval
        );
        assert_eq!(
            read_pixel_value(&17f32.to_le_bytes(), BandDataType::Float32).unwrap(),
            nodataval
        );
        assert_eq!(
            read_pixel_value(&17f64.to_le_bytes(), BandDataType::Float64).unwrap(),
            nodataval
        );

        let invalid = -300.0 as f64;
        assert!(read_pixel_value(&[invalid as u8], BandDataType::UInt8).is_err());
    }
}
