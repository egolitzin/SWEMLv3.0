"""
Compute land cover static attributes per HRU from NLCD.

Data source: NLCD 2021 via pygeohydro (web service)
Output: model_domain_{name}/landcover.parquet

Attributes computed:
    forest_fraction    - classes 41 (deciduous), 42 (evergreen), 43 (mixed)
    shrub_fraction     - class 52
    grass_fraction     - class 71
    impervious_fraction - class 21-24 (developed)
    dom_nlcd_class     - modal NLCD class

Requires: pip install pygeohydro exactextract

Usage:
    python landcover.py --domain sierra_nevada
    python landcover.py --domain sierra_nevada --year 2019
"""
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box
from domain_utils import domain_dir, load_hrus, domain_bbox

# NLCD class groups
FOREST_CLASSES    = [41, 42, 43]
SHRUB_CLASSES     = [52]
GRASS_CLASSES     = [71]
DEVELOPED_CLASSES = [21, 22, 23, 24]


def fetch_nlcd(bbox: tuple, year: int = 2021) -> xr.DataArray:
    try:
        from pygeohydro import nlcd_bygeom
    except ImportError:
        raise ImportError('pygeohydro required. Install with: pip install pygeohydro')
    import warnings
    lon_min, lat_min, lon_max, lat_max = bbox
    gs = gpd.GeoSeries([box(lon_min, lat_min, lon_max, lat_max)], crs='EPSG:4326')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning, module='pygeoutils')
        ds = nlcd_bygeom(gs, resolution=30, years={'cover': [year]})[0]
    return ds[f'cover_{year}']


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
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    out_file = domain_dir(args.domain) / 'landcover.parquet'
    if out_file.exists() and not args.overwrite:
        print('landcover.parquet already exists. Use --overwrite to redo.')
        return

    hrus = load_hrus(args.domain)
    bbox = domain_bbox(hrus)

    print(f'Fetching NLCD {args.year} at 30m...')
    nlcd = fetch_nlcd(bbox, args.year)

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
