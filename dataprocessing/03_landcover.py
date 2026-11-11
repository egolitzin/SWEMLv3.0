"""
Compute land cover static attributes per HRU from NLCD.

Data source: NLCD 2021 local raster (preferred) or MRLC WMS fallback
Output: model_domain_{name}/landcover.parquet

Attributes computed:
    forest_fraction    - classes 41 (deciduous), 42 (evergreen), 43 (mixed)
    shrub_fraction     - class 52
    grass_fraction     - class 71
    impervious_fraction - class 21-24 (developed)
    dom_nlcd_class     - modal NLCD class

Download NLCD 2021 CONUS raster once:
    wget -P /uufs/chpc.utah.edu/common/home/civil-group1/Golitzin/SWEMLv3.0/NLCD/ \
        https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/Annual_NLCD_LndCov_2021_CU_C1V1.zip
    unzip -d /uufs/chpc.utah.edu/common/home/civil-group1/Golitzin/SWEMLv3.0/NLCD/ \
        /uufs/chpc.utah.edu/common/home/civil-group1/Golitzin/SWEMLv3.0/NLCD/NLCD_2021_Land_Cover_L48_20230630.zip

Requires: pip install pygeohydro exactextract rioxarray

Usage:
    python landcover.py --domain sierra_nevada --nlcd-file /path/to/NLCD_2021_Land_Cover_L48_20230630.img
    python landcover.py --domain sierra_nevada --year 2019
"""
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
from pyproj import Transformer
from shapely.geometry import box
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from domain_utils import domain_dir, load_hrus, domain_bbox

# NLCD class groups
FOREST_CLASSES    = [41, 42, 43]
SHRUB_CLASSES     = [52]
GRASS_CLASSES     = [71]
DEVELOPED_CLASSES = [21, 22, 23, 24]


def _fetch_nlcd_local(nlcd_file: str, bbox: tuple) -> xr.DataArray:
    # bbox is (lon_min, lat_min, lon_max, lat_max) in EPSG:4326
    # open lazily first to read the raster's native CRS, then transform bbox into it.
    # all 4 corners are needed — in Albers the projected bbox of a geographic rectangle
    # is not defined by the SW/NE corners alone (NW gives y_max for WUS domains).
    da = rioxarray.open_rasterio(nlcd_file, chunks='auto').squeeze()
    t = Transformer.from_crs('EPSG:4326', da.rio.crs, always_xy=True)
    lon_min, lat_min, lon_max, lat_max = bbox
    xs, ys = t.transform([lon_min, lon_min, lon_max, lon_max],
                         [lat_min, lat_max, lat_min, lat_max])
    return da.rio.clip_box(min(xs), min(ys), max(xs), max(ys)).compute()


@retry(stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=30, min=30, max=300),
       retry=retry_if_exception_type(Exception))
def _fetch_nlcd_wms(bbox: tuple, year: int) -> xr.DataArray:
    from pygeohydro import nlcd_bygeom
    import warnings
    lon_min, lat_min, lon_max, lat_max = bbox
    gs = gpd.GeoSeries([box(lon_min, lat_min, lon_max, lat_max)], crs='epsg:4326')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning, module='pygeoutils')
        ds = nlcd_bygeom(gs, resolution=30, years={'cover': [year]}, crs='epsg:3857')[0]
    return ds[f'cover_{year}']


def fetch_nlcd(bbox: tuple, year: int = 2021, nlcd_file: str = None) -> xr.DataArray:
    if nlcd_file:
        print(f'Reading NLCD from local file: {nlcd_file}')
        return _fetch_nlcd_local(nlcd_file, bbox)
    print('Fetching NLCD from MRLC WMS (use --nlcd-file to avoid timeouts on large domains)...')
    return _fetch_nlcd_wms(bbox, year)


def class_fraction(nlcd: xr.DataArray, hrus: gpd.GeoDataFrame,
                   class_ids: list[int], col_name: str) -> pd.Series:
    """Fraction of pixels in class_ids within each HRU."""
    from exactextract import exact_extract

    # binary mask: 1 where pixel in class_ids, 0 elsewhere
    mask = xr.where(nlcd.isin(class_ids), 1.0, 0.0).rio.write_crs(nlcd.rio.crs)
    df   = exact_extract(mask, hrus.to_crs(nlcd.rio.crs), ['mean'],
                          include_cols=['divide_id'], output='pandas')
    return df.rename(columns={'mean': col_name})[['divide_id', col_name]]


def dominant_class(nlcd: xr.DataArray, hrus: gpd.GeoDataFrame) -> pd.Series:
    from exactextract import exact_extract
    df = exact_extract(nlcd.astype(float), hrus.to_crs(nlcd.rio.crs), ['mode'],
                        include_cols=['divide_id'], output='pandas')
    df['mode'] = df['mode'].round().astype('Int64')
    return df.rename(columns={'mode': 'dom_nlcd_class'})[['divide_id', 'dom_nlcd_class']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True)
    ap.add_argument('--year', type=int, default=2021, choices=[2016, 2019, 2021])
    ap.add_argument('--nlcd-file', default=None,
                    help='Path to local NLCD CONUS raster (avoids unreliable MRLC WMS)')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    out_file = domain_dir(args.domain) / 'landcover.parquet'
    if out_file.exists() and not args.overwrite:
        print('landcover.parquet already exists. Use --overwrite to redo.')
        return

    hrus = load_hrus(args.domain)
    bbox = domain_bbox(hrus)

    nlcd = fetch_nlcd(bbox, args.year, args.nlcd_file)

    print('Computing land cover fractions...')
    fracs = [
        class_fraction(nlcd, hrus, FOREST_CLASSES,    'forest_fraction'),
        class_fraction(nlcd, hrus, SHRUB_CLASSES,     'shrub_fraction'),
        class_fraction(nlcd, hrus, GRASS_CLASSES,     'grass_fraction'),
        class_fraction(nlcd, hrus, DEVELOPED_CLASSES, 'impervious_fraction'),
        dominant_class(nlcd, hrus),
    ]
    df = fracs[0]
    for f in fracs[1:]:
        df = df.merge(f, on='divide_id')

    df.to_parquet(out_file, index=False)
    print(f'Saved land cover attrs for {len(df)} HRUs → {out_file}')


if __name__ == '__main__':
    main()
