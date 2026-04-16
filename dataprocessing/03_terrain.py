"""
Compute terrain static attributes per HRU via DEM zonal statistics.

Data source: USGS 3DEP accessed via py3dep (no local files needed)
Output: model_domain_{name}/terrain.parquet

Attributes computed:
    elev_mean, elev_std, elev_min, elev_max, elev_p25, elev_p75
    slope_mean, aspect_sin, aspect_cos

Requires: pip install exactextract

Usage:
    python terrain.py --domain sierra_nevada
    python terrain.py --domain sierra_nevada --resolution 10
"""
import argparse
import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd
import py3dep
from domain_utils import domain_dir, load_hrus, domain_bbox

# EPSG:5070 (Albers Equal Area CONUS) gives metric pixel spacing everywhere
METRIC_CRS = 'EPSG:5070'


def fetch_dem(bbox: tuple, resolution: int = 30) -> xr.DataArray:
    return py3dep.get_dem(bbox, resolution=resolution, crs='EPSG:4326')


def compute_slope_aspect(dem: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Derive slope (deg) and aspect (deg, clockwise from north) from DEM.
    Reprojects to EPSG:5070 for metric coordinates, then uses
    xr.DataArray.differentiate so gradients respect actual coordinate sign
    regardless of raster storage order (N-to-S or S-to-N).
    """
    dem_m = dem.rio.reproject(METRIC_CRS).squeeze()

    dz_dy = dem_m.differentiate('y')   # dz/dy: positive = upslope going north
    dz_dx = dem_m.differentiate('x')   # dz/dx: positive = upslope going east

    slope_da  = np.rad2deg(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect_da = np.rad2deg(np.arctan2(-dz_dx, -dz_dy)) % 360  # clockwise from north

    slope_da  = slope_da.rio.write_crs(METRIC_CRS)
    aspect_da = aspect_da.rio.write_crs(METRIC_CRS)
    return slope_da, aspect_da


def zonal_stats(raster: xr.DataArray, hrus: gpd.GeoDataFrame,
                stats: list[str], prefix: str) -> pd.DataFrame:
    from exactextract import exact_extract
    rename = {
        'mean':             f'{prefix}_mean',
        'stdev':            f'{prefix}_std',
        'min':              f'{prefix}_min',
        'max':              f'{prefix}_max',
        'quantile(q=0.25)': f'{prefix}_p25',
        'quantile(q=0.75)': f'{prefix}_p75',
    }
    df = exact_extract(raster, hrus.to_crs(raster.rio.crs), stats,
                       include_cols=['divide_id'], output='pandas')
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True)
    ap.add_argument('--resolution', type=int, default=30,
                    help='DEM resolution in meters (default: 30)')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    out_file = domain_dir(args.domain) / 'terrain.parquet'
    if out_file.exists() and not args.overwrite:
        print('terrain.parquet already exists. Use --overwrite to redo.')
        return

    hrus = load_hrus(args.domain)
    bbox = domain_bbox(hrus)

    print(f'Fetching {args.resolution}m DEM from USGS 3DEP...')
    dem = fetch_dem(bbox, args.resolution)

    print('Computing slope and aspect in EPSG:5070...')
    slope, aspect = compute_slope_aspect(dem)

    # elevation stats stay in EPSG:4326 (matches DEM source); slope/aspect in EPSG:5070
    hrus_4326  = hrus.to_crs('EPSG:4326')
    hrus_5070  = hrus.to_crs(METRIC_CRS)

    print('Running zonal statistics...')
    elev_stats  = zonal_stats(dem,    hrus_4326,
                               ['mean', 'stdev', 'min', 'max',
                                'quantile(q=0.25)', 'quantile(q=0.75)'], 'elev')
    slope_stats  = zonal_stats(slope,  hrus_5070, ['mean'],  'slope')
    aspect_stats = zonal_stats(aspect, hrus_5070, ['mean'],  'aspect')

    aspect_rad = np.deg2rad(aspect)
    aspect_sin_r = np.sin(aspect_rad).rio.write_crs(METRIC_CRS)
    aspect_cos_r = np.cos(aspect_rad).rio.write_crs(METRIC_CRS)
    sin_stats = zonal_stats(aspect_sin_r, hrus_5070, ['mean'], 'aspect_sin')
    cos_stats = zonal_stats(aspect_cos_r, hrus_5070, ['mean'], 'aspect_cos')

    df = (elev_stats
          .merge(slope_stats[['divide_id', 'slope_mean']],     on='divide_id')
          .merge(aspect_stats[['divide_id', 'aspect_mean']],   on='divide_id')
          .merge(sin_stats[['divide_id', 'aspect_sin_mean']],  on='divide_id')
          .merge(cos_stats[['divide_id', 'aspect_cos_mean']],  on='divide_id'))

    df = df.rename(columns={'aspect_sin_mean': 'aspect_sin', 'aspect_cos_mean': 'aspect_cos'})
    df['aspect_mean']=(np.arctan2(df['aspect_sin'],df['aspect_cos'])*180/np.pi)%360

    df.to_parquet(out_file, index=False)
    print(f'Saved terrain attrs for {len(df)} HRUs → {out_file}')


if __name__ == '__main__':
    main()
