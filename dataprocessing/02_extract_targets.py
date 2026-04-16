"""
Extract UCLA SWE posterior mean per HRU and save as zarr timeseries.

Data source: WUS_UCLA_SR v01, ~500m resolution
    Tiles: WUS_UCLA_SR_v01_N{lat}_0W{lon}_0_agg_16_WY{wy}_{yy}_SWE_SCA_POST.nc
    Dims: (Day:365, Stats:5, Longitude:225, Latitude:225)
    Stats=0: posterior mean SWE [mm]
    Latitude coordinate: N to S (e.g. 32.0 -> 31.0 for tile N31_0)
    Longitude coordinate: W to E (e.g. -103.0 -> -102.0 for tile W103_0)

Tile naming: lat/lon are the LOWER-LEFT (SW) corner of the 1-degree x 1-degree tile.
    N{lat}_0W{lon}_0: covers [lat, lat+1]N and [-lon, -(lon-1)] longitude.

Output: model_domain_{name}/targets.zarr  (time, divide_id), variable swe [mm]
        model_domain_{name}/snow_hrus.txt  divide_ids with meaningful snow signal
        model_domain_{name}/hrus.parquet   overwritten to snow-present HRUs only

Run before static attribute steps so they only process relevant HRUs.

Usage:
    python extract_targets.py --domain sierra_nevada
    python extract_targets.py --domain sierra_nevada --wy_start 2014 --wy_end 2020
"""
import argparse
import shutil
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from domain_utils import domain_dir, load_initial_hrus, domain_bbox, BASE_LOC

UCLA_DIR = BASE_LOC / 'UCLA_SWE'
PPD      = 225       # pixels per degree in UCLA tiles
STATS_I  = 0         # index 0 = posterior mean SWE

# Snow filter thresholds
MIN_SWE_MM    = 10.0
MIN_SNOW_DAYS = 7
MIN_SNOW_WYS  = 1


# ---------------------------------------------------------------------------
# Tile utilities
# ---------------------------------------------------------------------------

def tile_name(lat_s: int, lon_w: int, wy: int) -> str:
    """Filename for tile covering water year wy.
    """
    yy = str(wy)[2:]   # last two digits of the ending year
    return f'WUS_UCLA_SR_v01_N{lat_s}_0W{lon_w}_0_agg_16_WY{wy - 1}_{yy}_SWE_SCA_POST.nc'


def _parse_tile_coord(fname: str) -> tuple[int, int, int] | None:
    """Parse (lat_s, lon_w, wy) from a UCLA SWE filename, or None if no match."""
    import re
    m = re.search(r'N(\d+)_0W(\d+)_0_agg_16_WY(\d{4})_(\d{2})_', fname)
    if not m:
        return None
    lat_s, lon_w = int(m.group(1)), int(m.group(2))
    wy = int(m.group(3)[:2] + m.group(4))   # e.g. "20" + "15" -> 2015
    return lat_s, lon_w, wy


def _available_tiles() -> dict[tuple[int, int, int], Path]:
    """Scan UCLA_DIR once and return {(lat_s, lon_w, wy): path} for all tiles."""
    index = {}
    for f in UCLA_DIR.glob('WUS_UCLA_SR_v01_*.nc'):
        coord = _parse_tile_coord(f.name)
        if coord is not None:
            index[coord] = f
    return index


# Module-level cache so we only scan the directory once per process
_TILE_INDEX: dict | None = None


def tiles_for_bbox(bbox: tuple, wy: int) -> list[tuple[int, int, Path]]:
    """
    Return (lat_s, lon_w, path) for every tile that exists on disk and
    overlaps bbox in water year wy. Tiles are discovered by scanning UCLA_DIR,
    not by enumerating a rectangular grid, because coverage is irregular.
    """
    global _TILE_INDEX
    if _TILE_INDEX is None:
        _TILE_INDEX = _available_tiles()

    lon_min, lat_min, lon_max, lat_max = bbox

    tiles = []
    for (lat_s, lon_w, tile_wy), fpath in _TILE_INDEX.items():
        if tile_wy != wy:
            continue
        # tile covers [lat_s, lat_s+1] N and [-lon_w, -(lon_w-1)] lon
        tile_lon_min = -lon_w
        tile_lon_max = -(lon_w - 1)
        tile_lat_min =  lat_s
        tile_lat_max =  lat_s + 1
        if tile_lon_min < lon_max and tile_lon_max > lon_min \
                and tile_lat_min < lat_max and tile_lat_max > lat_min:
            tiles.append((lat_s, lon_w, fpath))
    return tiles


def wy_dates(wy: int) -> pd.DatetimeIndex:
    """365 daily dates for water year wy: Oct 1 of (wy-1) through Sep 30 of wy."""
    return pd.date_range(pd.Timestamp(wy - 1, 10, 1), periods=365, freq='D')


# ---------------------------------------------------------------------------
# Mosaic grid and HRU rasterization
# ---------------------------------------------------------------------------

def mosaic_coords(lat_ss: list[int], lon_ws: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build lat/lon pixel-center coordinate arrays for a mosaic of tiles.
    lat_ss: sorted list of tile southern-edge latitudes
    lon_ws: sorted list of tile western-edge abs-longitudes
    Returns (lat_coords [N to S], lon_coords [W to E]).
    """
    lat_coords = []
    for lat_s in sorted(lat_ss, reverse=True):   # north to south
        centers = lat_s + 1.0 - (np.arange(PPD) + 0.5) / PPD
        lat_coords.append(centers)

    lon_coords = []
    for lon_w in sorted(lon_ws, reverse=True):    # west to east (largest lon_w = westernmost)
        centers = -lon_w + (np.arange(PPD) + 0.5) / PPD
        lon_coords.append(centers)

    return np.concatenate(lat_coords), np.concatenate(lon_coords)


def rasterize_hrus(hrus, lat_coords: np.ndarray, lon_coords: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Rasterize HRUs onto the mosaic grid.
    Returns:
        mask: (n_lat, n_lon) int32 array, 1-based HRU index (0 = no HRU)
        hru_ids: array of divide_id strings corresponding to mask values 1..N
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    n_lat = len(lat_coords)
    n_lon = len(lon_coords)
    dy = lat_coords[0] - lat_coords[1]   # positive: lat decreasing downward
    dx = lon_coords[1] - lon_coords[0]   # positive: lon increasing rightward

    west  = lon_coords[0]  - dx / 2
    east  = lon_coords[-1] + dx / 2
    north = lat_coords[0]  + dy / 2
    south = lat_coords[-1] - dy / 2
    transform = from_bounds(west, south, east, north, n_lon, n_lat)

    hrus_4326 = hrus.to_crs('EPSG:4326').reset_index(drop=True)
    shapes = [(geom, idx + 1) for idx, geom in enumerate(hrus_4326.geometry)]

    mask = rasterize(
        shapes,
        out_shape=(n_lat, n_lon),
        transform=transform,
        fill=0,
        dtype='int32',
    )
    return mask, hrus_4326['divide_id'].values


# ---------------------------------------------------------------------------
# Per-tile SWE extraction
# ---------------------------------------------------------------------------

def load_tile_swe(fpath: Path) -> np.ndarray:
    """
    Load SWE_Post[Stats=0] from one tile.
    Returns float32 array shaped (365, PPD, PPD) with lat N-to-S, lon W-to-E.
    Fill values (< -9000) set to NaN.
    """
    ds  = xr.open_dataset(fpath)
    swe = ds['SWE_Post'].isel(Stats=STATS_I).values  # (365, Longitude=225, Latitude=225)
    ds.close()

    swe = swe.transpose(0, 2, 1).astype(np.float32)  # (365, Latitude, Longitude)
    # Latitude coordinate in file is N-to-S (32.0 -> 31.0), so row order is already correct.
    swe[swe < -9000] = np.nan
    swe *= 1000.0   # convert m -> mm
    return swe  # (365, PPD, PPD)


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_swe(domain: str, wy_start: int, wy_end: int) -> xr.Dataset:
    hrus = load_initial_hrus(domain)
    bbox = domain_bbox(hrus)

    # Determine mosaic grid from tiles available in any WY (grid is static)
    ref_tiles = tiles_for_bbox(bbox, wy_start)
    if not ref_tiles:
        raise RuntimeError(
            f'No UCLA SWE tiles found for WY{wy_start} in {UCLA_DIR}.\n'
            f'Run utils/get_ucla_swe.py first.'
        )

    lat_ss = sorted(set(t[0] for t in ref_tiles))
    lon_ws = sorted(set(t[1] for t in ref_tiles))
    lat_coords, lon_coords = mosaic_coords(lat_ss, lon_ws)
    n_lat = len(lat_coords)
    n_lon = len(lon_coords)

    print(f'  Mosaic: {n_lat} x {n_lon} pixels '
          f'({lat_coords[-1]:.2f}–{lat_coords[0]:.2f} N, '
          f'{lon_coords[0]:.2f}–{lon_coords[-1]:.2f} E)')

    print('  Rasterizing HRUs...')
    mask, hru_ids = rasterize_hrus(hrus, lat_coords, lon_coords)
    n_hrus = len(hru_ids)
    print(f'  {n_hrus} HRUs, {(mask > 0).sum()} rasterized pixels')

    # Precompute tile placement in mosaic
    lat_ss_desc  = sorted(lat_ss, reverse=True)
    lon_ws_desc  = sorted(lon_ws, reverse=True)   # west to east, matching mosaic_coords
    lat_s_to_row = {ls: i * PPD for i, ls in enumerate(lat_ss_desc)}
    lon_w_to_col = {lw: i * PPD for i, lw in enumerate(lon_ws_desc)}

    all_swe   = []
    all_dates = []

    for wy in range(wy_start, wy_end + 1):
        print(f'  WY{wy}...')
        tiles = tiles_for_bbox(bbox, wy)
        if not tiles:
            print(f'    No tiles found, skipping WY{wy}.')
            continue

        sums   = np.zeros((365, n_hrus), dtype=np.float64)
        counts = np.zeros((365, n_hrus), dtype=np.float64)

        for lat_s, lon_w, fpath in tiles:
            r0 = lat_s_to_row.get(lat_s)
            c0 = lon_w_to_col.get(lon_w)
            if r0 is None or c0 is None:
                continue

            tile_swe  = load_tile_swe(fpath)           # (365, PPD, PPD)
            tile_mask = mask[r0:r0+PPD, c0:c0+PPD]    # (PPD, PPD)
            flat_mask = tile_mask.ravel()
            has_hru   = flat_mask > 0                  # (PPD*PPD,)

            for day_i in range(365):
                flat_swe = tile_swe[day_i].ravel()
                valid    = has_hru & ~np.isnan(flat_swe)
                if not valid.any():
                    continue
                idx = flat_mask[valid]   # 1-based
                sums[day_i]   += np.bincount(idx, weights=flat_swe[valid],
                                             minlength=n_hrus + 1)[1:]
                counts[day_i] += np.bincount(idx, minlength=n_hrus + 1)[1:]

        with np.errstate(invalid='ignore', divide='ignore'):
            mean_swe = np.where(counts > 0, sums / counts, np.nan).astype(np.float32)
        n_valid = np.sum(counts > 0)
        peak    = float(np.nanmax(mean_swe)) if n_valid > 0 else 0.0
        print(f'    {n_valid} (day, HRU) pairs with data, peak SWE = {peak:.1f} mm')
        all_swe.append(mean_swe)
        all_dates.append(wy_dates(wy).values)

    all_data  = np.concatenate(all_swe,   axis=0)  # (total_days, n_hrus)
    all_times = np.concatenate(all_dates, axis=0)

    return xr.Dataset(
        {'swe': xr.DataArray(all_data, dims=['time', 'divide_id'],
                             attrs={'units': 'mm', 'long_name': 'SWE posterior mean'})},
        coords={'time': all_times, 'divide_id': hru_ids},
    )


def filter_snow_hrus(ds: xr.Dataset) -> list[str]:
    """Return divide_ids with at least MIN_SNOW_DAYS > MIN_SWE_MM in >= MIN_SNOW_WYS years."""
    swe    = ds['swe'].values                                        # (time, n_hrus)
    times  = pd.DatetimeIndex(ds['time'].values)
    wy_arr = np.where(times.month >= 10, times.year + 1, times.year)

    hru_ids   = ds['divide_id'].values
    snow_wys  = np.zeros(len(hru_ids), dtype=int)

    for wy_val in np.unique(wy_arr):
        mask_wy     = wy_arr == wy_val
        days_snow   = np.nansum(swe[mask_wy] > MIN_SWE_MM, axis=0)  # (n_hrus,)
        snow_wys   += (days_snow >= MIN_SNOW_DAYS).astype(int)

    return list(hru_ids[snow_wys >= MIN_SNOW_WYS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain',   required=True)
    ap.add_argument('--wy_start', type=int, default=2015)
    ap.add_argument('--wy_end',   type=int, default=2021)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    out_dir  = domain_dir(args.domain)
    out_zarr = out_dir / 'targets.zarr'
    out_hrus = out_dir / 'snow_hrus.txt'

    if out_zarr.exists() and not args.overwrite:
        print('targets.zarr already exists. Use --overwrite to redo.')
        return

    print(f'Extracting UCLA SWE for domain "{args.domain}" '
          f'WY{args.wy_start}–{args.wy_end}...')
    ds = extract_swe(args.domain, args.wy_start, args.wy_end)

    n_days = ds.sizes['time']
    n_hrus = ds.sizes['divide_id']
    print(f'Saving targets.zarr ({n_days} days x {n_hrus} HRUs)...')
    if out_zarr.exists():
        shutil.rmtree(out_zarr)
    ds.to_zarr(out_zarr)

    print('Filtering HRUs by snow presence...')
    snow_ids = filter_snow_hrus(ds)
    n_snow   = len(snow_ids)
    print(f'  {n_snow} / {n_hrus} HRUs have snow signal')
    out_hrus.write_text('\n'.join(snow_ids) + '\n')
    print(f'  Saved snow HRU list -> {out_hrus}')

    # Overwrite hrus.parquet with snow-present HRUs only so all downstream
    # steps (02_terrain, 02_landcover, etc.) only process relevant HRUs.
    import geopandas as gpd
    hrus_full = gpd.read_parquet(out_dir / 'initial_hrus.parquet')
    hrus_snow = hrus_full[hrus_full['divide_id'].isin(set(snow_ids))].copy()
    hrus_snow.to_parquet(out_dir / 'hrus.parquet')
    print(f'  Updated hrus.parquet: {n_snow} snow HRUs (was {n_hrus})')

    # Trim targets.zarr to snow HRUs only
    ds_snow = ds.sel(divide_id=snow_ids)
    shutil.rmtree(out_zarr)
    ds_snow.to_zarr(out_zarr)
    print(f'  Trimmed targets.zarr to {n_snow} snow HRUs')

    t0   = str(ds.time.values[0])[:10]
    t1   = str(ds.time.values[-1])[:10]
    peak = float(np.nanmax(ds_snow['swe'].values))
    print(f'\nSWE summary:')
    print(f'  Time range: {t0} to {t1}')
    print(f'  Peak SWE across snow HRUs: {peak:.1f} mm')


if __name__ == '__main__':
    main()
