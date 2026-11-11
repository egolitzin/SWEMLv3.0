"""
Compute terrain static attributes per HRU via DEM zonal statistics.

Data source: USGS 3DEP accessed via py3dep
Output: model_domain_{name}/terrain.parquet

Attributes computed:
    elev_mean, elev_std, elev_min, elev_max, elev_p25, elev_p75
    relief, hypsometric_integral, elev_p25_norm, elev_p75_norm
    slope_mean, slope_std, aspect_sin, aspect_cos

The domain is processed in spatial tiles (--tile-deg, default 5°) so that
large or geographically dispersed domains (e.g. multi-state ASO domain) never
require a single DEM covering the full bounding box. Each HRU is assigned to
the tile containing its centroid; the DEM is fetched with a small buffer so
edge HRUs are fully covered.

Requires: pip install exactextract

Usage:
    python terrain.py --domain sierra_nevada
    python terrain.py --domain aso --tile-deg 3
"""
import argparse
import time
import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd
import py3dep
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from domain_utils import domain_dir, load_hrus, domain_bbox

# EPSG:5070 (Albers Equal Area CONUS) gives metric pixel spacing everywhere
METRIC_CRS = 'EPSG:5070'


@retry(stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=60, min=60, max=600),
       retry=retry_if_exception_type(Exception))
def fetch_dem(bbox: tuple, resolution: int = 90) -> xr.DataArray:
    return py3dep.get_dem(bbox, resolution=resolution, crs='EPSG:4326').squeeze()


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
    del dem_m

    slope_da  = np.rad2deg(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect_da = np.rad2deg(np.arctan2(-dz_dx, -dz_dy)) % 360  # clockwise from north
    del dz_dx, dz_dy

    slope_da  = slope_da.rio.write_crs(METRIC_CRS)
    aspect_da = aspect_da.rio.write_crs(METRIC_CRS)
    return slope_da, aspect_da


def zonal_stats(raster: xr.DataArray, hrus: gpd.GeoDataFrame,
                stats: list[str], prefix: str) -> pd.DataFrame:
    from exactextract import exact_extract
    rename = {
        'mean':        f'{prefix}_mean',
        'stdev':       f'{prefix}_std',
        'min':         f'{prefix}_min',
        'max':         f'{prefix}_max',
        'quantile_25': f'{prefix}_p25',
        'quantile_75': f'{prefix}_p75',
    }
    df = exact_extract(raster, hrus.to_crs(raster.rio.crs), stats,
                       include_cols=['divide_id'], output='pandas')
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def process_tile(hrus_4326: gpd.GeoDataFrame, dem_bbox: tuple,
                 resolution: int) -> pd.DataFrame | None:
    """Fetch DEM for one tile and compute terrain stats for HRUs in that tile."""
    try:
        dem = fetch_dem(dem_bbox, resolution)

        slope, aspect = compute_slope_aspect(dem)

        elev_stats = zonal_stats(dem, hrus_4326,
                                 ['mean', 'stdev', 'min', 'max',
                                  'quantile(q=0.25)', 'quantile(q=0.75)'], 'elev')
        del dem

        slope_stats = zonal_stats(slope, hrus_4326, ['mean', 'stdev'], 'slope')
        del slope

        aspect_rad = np.deg2rad(aspect)
        del aspect

        sin_stats = zonal_stats(np.sin(aspect_rad).rio.write_crs(METRIC_CRS),
                                hrus_4326, ['mean'], 'aspect_sin')
        cos_stats = zonal_stats(np.cos(aspect_rad).rio.write_crs(METRIC_CRS),
                                hrus_4326, ['mean'], 'aspect_cos')
        del aspect_rad

        return (elev_stats
                .merge(slope_stats[['divide_id', 'slope_mean', 'slope_std']],  on='divide_id')
                .merge(sin_stats[['divide_id', 'aspect_sin_mean']],            on='divide_id')
                .merge(cos_stats[['divide_id', 'aspect_cos_mean']],            on='divide_id'))

    except Exception as e:
        print(f'    Tile failed: {e}')
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True)
    ap.add_argument('--resolution', type=int, default=90,
                    help='DEM resolution in meters (default: 90)')
    ap.add_argument('--tile-deg', type=float, default=5.0,
                    help='Tile size in degrees (default: 5.0). Smaller values use '
                         'less memory; 5° tiles are ~25M pixels at 90m.')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    out_file = domain_dir(args.domain) / 'terrain.parquet'
    if out_file.exists() and not args.overwrite:
        print('terrain.parquet already exists. Use --overwrite to redo.')
        return

    hrus     = load_hrus(args.domain)
    hrus_4326 = hrus.to_crs('EPSG:4326')
    bbox     = domain_bbox(hrus)

    # Build tile grid aligned to --tile-deg
    BUFFER = 0.1   # degrees added to each tile's DEM fetch to cover edge HRUs
    t = args.tile_deg
    lon_min, lat_min, lon_max, lat_max = bbox
    lon_edges = np.arange(np.floor(lon_min / t) * t, lon_max + t, t)
    lat_edges = np.arange(np.floor(lat_min / t) * t, lat_max + t, t)

    # Assign each HRU to the tile containing its centroid.
    # Centroid must be computed in a projected CRS; convert back to lon/lat for tile indexing.
    centroids = hrus.to_crs(METRIC_CRS).geometry.centroid.to_crs('EPSG:4326')
    cx = centroids.x.values
    cy = centroids.y.values
    tile_col = np.clip(np.digitize(cx, lon_edges) - 1, 0, len(lon_edges) - 2)
    tile_row = np.clip(np.digitize(cy, lat_edges) - 1, 0, len(lat_edges) - 2)

    occupied = np.unique(np.column_stack([tile_row, tile_col]), axis=0)
    print(f'{len(hrus)} HRUs across {len(occupied)} tiles '
          f'({t}° grid, {args.resolution}m DEM)...')

    def run_tile(mask, label):
        """Process one set of HRUs (boolean mask into hrus_4326). Returns df or None."""
        b = hrus_4326[mask].total_bounds
        bbox = (b[0] - BUFFER, b[1] - BUFFER, b[2] + BUFFER, b[3] + BUFFER)
        print(f'  {label}: {mask.sum()} HRUs  '
              f'bbox=({bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f})')
        time.sleep(5)
        return process_tile(hrus_4326[mask], bbox, args.resolution)

    all_dfs = []
    for i, (row, col) in enumerate(occupied):
        idx = (tile_row == row) & (tile_col == col)

        df_tile = run_tile(idx, f'Tile {i + 1}/{len(occupied)}')
        if df_tile is not None:
            all_dfs.append(df_tile)
            continue

        # Tile failed — split along the longer bbox axis and retry each half.
        # This halves the pixel count, handling cases where the bbox is near
        # py3dep's internal MAX_PIXELS limit.
        print('    Splitting into sub-tiles and retrying...')
        b = hrus_4326[idx].total_bounds
        sub = np.where(idx)[0]
        split_by_lon = (b[2] - b[0]) >= (b[3] - b[1])
        order = np.argsort(cx[sub] if split_by_lon else cy[sub])
        mid = len(order) // 2
        for k, pos in enumerate([order[:mid], order[mid:]]):
            half = np.zeros(len(hrus), dtype=bool)
            half[sub[pos]] = True
            df_half = run_tile(half, f'  Sub-tile {i + 1}{chr(ord("a") + k)}')
            if df_half is not None:
                all_dfs.append(df_half)

    if not all_dfs:
        print('No tiles processed successfully.')
        return

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.rename(columns={'aspect_sin_mean': 'aspect_sin',
                             'aspect_cos_mean': 'aspect_cos'})
    df['aspect_mean'] = (np.arctan2(df['aspect_sin'], df['aspect_cos']) * 180 / np.pi) % 360

    elev_range = df['elev_max'] - df['elev_min']
    df['relief']               = elev_range
    df['hypsometric_integral'] = (df['elev_mean'] - df['elev_min']) / elev_range
    df['elev_p25_norm']        = (df['elev_p25']  - df['elev_min']) / elev_range
    df['elev_p75_norm']        = (df['elev_p75']  - df['elev_min']) / elev_range
    for col in ['hypsometric_integral', 'elev_p25_norm', 'elev_p75_norm']:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    low_relief = df[df['relief'] < 50]
    if len(low_relief):
        print(f'  Note: {len(low_relief)} HRUs with relief < 50m — normalized elevation features may be unreliable for these basins.')

    n_missing = len(hrus) - len(df)
    if n_missing:
        print(f'Warning: {n_missing} HRUs missing (tile failed entirely).')
    n_nan = df.drop(columns='divide_id').isna().any(axis=1).sum()
    if n_nan:
        print(f'Warning: {n_nan} HRUs have NaN attributes (edge coverage gap).')

    df.to_parquet(out_file, index=False)
    print(f'Saved terrain attrs for {len(df)} HRUs → {out_file}')


if __name__ == '__main__':
    main()
