
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
        return Err(ArrowError::InvalidArgumentError("Invalid byte length for specified data type".to_string()));
    }

    match data_type {
        BandDataType::UInt8 => {
            Ok(bytes[0] as f64)
        }
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
                bytes[0], bytes[1], bytes[2], bytes[3],
                bytes[4], bytes[5], bytes[6], bytes[7]
            ]);
            Ok(value)
        }
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
}