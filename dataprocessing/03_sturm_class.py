"""
Assign Sturm snow climate classification to each HRU.

Data source: NSIDC-0768, North America 300m GeoTIFF
    https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0768_global_seasonal_snow_classification_v01/
    Requires Earthdata login (credentials in ~/.netrc)
Output: model_domain_{name}/sturm.parquet

Classes (integer codes in the raster):
    1 = tundra
    2 = boreal forest (taiga)
    3 = maritime
    4 = ephemeral
    5 = prairie
    6 = montane forest
    7 = ice
    

Attribute saved: sturm_class (integer modal class per HRU)

Requires: pip install exactextract earthaccess

Usage:
    python sturm_class.py --domain sierra_nevada
"""
import argparse
import earthaccess
from pathlib import Path
from domain_utils import domain_dir, load_hrus, domain_bbox

STURM_URL = (
    'https://daacdata.apps.nsidc.org/pub/DATASETS/'
    'nsidc0768_global_seasonal_snow_classification_v01/'
    'SnowClass_NA_300m_10.0arcsec_2021_v01.0.tif'
)

STURM_LABELS = {
    1:'tundra',
    2:'boreal_forest',
    3:'maritime',
    4:'ephemeral',
    5:'prairie',
    6:'montane_forest',
    7:'ice',
    8:'ocean',
    9:'fill'
}


def download_sturm_raster(cache_dir: Path) -> Path:
    """Download Sturm classification GeoTIFF to cache_dir if not already present."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / Path(STURM_URL).name
    if dest.exists():
        print(f'Using cached Sturm raster: {dest}')
        return dest

    earthaccess.login(strategy='netrc')
    session = earthaccess.get_requests_https_session()
    print('Downloading Sturm raster from NSIDC...')
    with session.get(STURM_URL, stream=True) as r:
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    out_file = domain_dir(args.domain) / 'sturm.parquet'
    if out_file.exists() and not args.overwrite:
        print('sturm.parquet already exists. Use --overwrite to redo.')
        return

    hrus  = load_hrus(args.domain)
    cache = domain_dir(args.domain) / '_cache' / 'sturm'

    print('Downloading Sturm classification raster (NSIDC-0768)...')
    raster_path = download_sturm_raster(cache)

    print('Computing modal Sturm class per HRU...')
    import rioxarray as rxr
    from exactextract import exact_extract

    rast = rxr.open_rasterio(raster_path, masked=True).squeeze()
    bbox = domain_bbox(hrus)
    rast = rast.rio.clip_box(*bbox, crs='EPSG:4326')
    df   = exact_extract(rast.astype(float), hrus.to_crs(rast.rio.crs), ['mode'],
                          include_cols=['divide_id'], output='pandas')
    df['sturm_class']      = df['mode'].round().astype('Int64')
    df['sturm_class_name'] = df['sturm_class'].map(STURM_LABELS)
    df = df.drop(columns=['mode'])

    df.to_parquet(out_file, index=False)
    print(f'Saved Sturm class for {len(df)} HRUs → {out_file}')
    print(df['sturm_class_name'].value_counts().to_string())


if __name__ == '__main__':
    main()
