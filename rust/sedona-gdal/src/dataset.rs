use arrow_schema::ArrowError;
use gdal::{Dataset};
use sedona_schema::datatypes::{BandMetadataRef, StorageType};

/// Get the out-db dataset reference from a raster band.
pub fn get_outdb_dataset(metadata: &dyn BandMetadataRef) -> Result<Dataset, ArrowError> {
    if metadata.storage_type() != StorageType::OutDbRef {
        return Err(ArrowError::ParseError(
            "Raster band is not stored out-of-db".to_string(),
        ));
    }

    let url = match metadata.outdb_url() {
        Some(url) => url,
        None => {
            return Err(ArrowError::ParseError(
                "Raster band does not have an out-db URL".to_string(),
            ))
        }
    };


    // These datasets may appear in multiple rasters and called repeatedly.
    // Adding a caching layer here would improve performance.
    open_outdb_band(&url)
}

fn open_outdb_band(url: &str) -> Result<Dataset, ArrowError> {
    let full_url = format!("/vsicurl/{}", url);
    let ds = Dataset::open(full_url).map_err(|e| ArrowError::ParseError(e.to_string()))?;
    Ok(ds)
}