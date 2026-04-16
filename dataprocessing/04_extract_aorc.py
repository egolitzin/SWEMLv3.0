"""
Extract AORC daily forcings per HRU via pixel-weighted zonal means.

Data source: s3://noaa-nws-aorc-v1-1-1km/{year}.zarr (public, no credentials)
    dims: (time, latitude, longitude), EPSG:4326, 1-km resolution
    variables: APCP_surface, TMP_2maboveground, SPFH_2maboveground, PRES_surface,
               DSWRF_surface, DLWRF_surface, UGRD_10maboveground, VGRD_10maboveground

Derived variables added before extraction:
    SNOW_wetbulb, RAIN_wetbulb  - phase partition via Stull (2011) wet-bulb

Output: model_domain_{name}/aorc.zarr  (time, divide_id)

Each WY spans two calendar years. Both are streamed from S3 in parallel via
ThreadPoolExecutor. Domain subset is loaded into memory (~1-2 GB per year at
1 km), then zonal means are computed and the subset is discarded.

Usage:
    python 04_extract_aorc.py --domain sierra_nevada
    python 04_extract_aorc.py --domain sierra_nevada --wy_start 2015 --wy_end 2021
"""
import argparse
import shutil
import numpy as np
import pandas as pd
import xarray as xr
import s3fs
from concurrent.futures import ThreadPoolExecutor
from scipy.sparse import csr_matrix
from domain_utils import domain_dir, load_hrus, domain_bbox

AORC_S3_BUCKET = 'noaa-nws-aorc-v1-1-1km'

MEAN_VARS = [
    'TMP_2maboveground', 'SPFH_2maboveground', 'PRES_surface',
    'DSWRF_surface', 'DLWRF_surface',
    'UGRD_10maboveground', 'VGRD_10maboveground',
]
SUM_VARS = ['APCP_surface']
OUT_VARS = SUM_VARS + MEAN_VARS + ['SNOW_wetbulb', 'RAIN_wetbulb']

T_ALL_SNOW = 0.0
T_ALL_RAIN = 2.0


def add_snow_rain(ds: xr.Dataset) -> xr.Dataset:
    """Add SNOW_wetbulb and RAIN_wetbulb via Stull (2011) wet-bulb linear ramp."""
    T_K   = ds['TMP_2maboveground']
    P_hPa = ds['PRES_surface'] / 100.0
    q     = ds['SPFH_2maboveground']
    T_C   = T_K - 273.15

    es = 6.112 * np.exp(17.67 * T_C / (T_C + 243.5))
    e  = q * P_hPa / (0.622 + 0.378 * q)
    RH = (e / es * 100.0).clip(0.0, 100.0)

    Tw = (T_C * np.arctan(0.151977 * (RH + 8.313659) ** 0.5)
          + np.arctan(T_C + RH)
          - np.arctan(RH - 1.676331)
          + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH)
          - 4.686035)

    P_snow = ((T_ALL_RAIN - Tw) / (T_ALL_RAIN - T_ALL_SNOW)).clip(0.0, 1.0)
    ds['SNOW_wetbulb'] = ds['APCP_surface'] * P_snow
    ds['RAIN_wetbulb'] = ds['APCP_surface'] * (1.0 - P_snow)
    return ds


def rasterize_hrus(hrus, lat: np.ndarray, lon: np.ndarray):
    """
    Map HRU polygons onto a regular lat/lon grid.
    lat: 1D ascending (S-to-N), lon: 1D ascending (W-to-E).
    Returns mask (n_lat, n_lon) int32, 1-based HRU index (0 = no HRU),
    and hru_ids array.
    """
    from rasterio.features import rasterize as rio_rast
    from rasterio.transform import from_bounds

    n_lat, n_lon = len(lat), len(lon)
    dlat = float(lat[1] - lat[0])
    dlon = float(lon[1] - lon[0])
    west  = float(lon[0])  - dlon / 2
    east  = float(lon[-1]) + dlon / 2
    south = float(lat[0])  - dlat / 2
    north = float(lat[-1]) + dlat / 2

    transform = from_bounds(west, south, east, north, n_lon, n_lat)
    hrus_4326 = hrus.to_crs('EPSG:4326').reset_index(drop=True)
    shapes = [(geom, i + 1) for i, geom in enumerate(hrus_4326.geometry)]
    mask_ns = rio_rast(shapes, out_shape=(n_lat, n_lon),
                       transform=transform, fill=0, dtype='int32')
    return mask_ns[::-1, :], hrus_4326['divide_id'].values


def build_weight_matrix(mask: np.ndarray, n_hrus: int) -> csr_matrix:
    """
    Build sparse (n_pixels, n_hrus) matrix S with 1.0 where pixel p is in HRU h.
    mask: (n_lat, n_lon) int32, 1-based HRU index (0 = no HRU).
    """
    flat = mask.ravel()
    pix  = np.where(flat > 0)[0]
    hru  = flat[pix] - 1
    return csr_matrix((np.ones(len(pix)), (pix, hru)), shape=(len(flat), n_hrus))


def zonal_means(data: np.ndarray, S: csr_matrix) -> np.ndarray:
    """
    Pixel-weighted mean per HRU for all time steps via sparse matrix multiply.
    data : (n_time, n_lat, n_lon)
    S    : (n_pixels, n_hrus) sparse, built by build_weight_matrix
    Returns (n_time, n_hrus) float32, NaN where no valid pixels.
    """
    flat = data.reshape(len(data), -1).astype(np.float64)
    valid_data = np.where(np.isfinite(flat), flat, 0.0)
    count_data = np.isfinite(flat).astype(np.float64)
    sums   = S.T.dot(valid_data.T).T
    counts = S.T.dot(count_data.T).T
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(counts > 0, sums / counts, np.nan).astype(np.float32)


def load_aorc_year(year: int, bbox: tuple, s3: s3fs.S3FileSystem) -> xr.Dataset:
    """
    Stream one calendar year of AORC from S3, clip to bbox, aggregate to daily,
    add snow/rain partition, and load into memory.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    store = s3fs.S3Map(root=f'{AORC_S3_BUCKET}/{year}.zarr', s3=s3, check=False)
    ds = xr.open_zarr(store, consolidated=True, chunks={})
    ds = ds.sel(
        latitude=slice(lat_min, lat_max),
        longitude=slice(lon_min, lon_max),
    )
    ds_daily = xr.merge([
        ds[SUM_VARS].resample(time='1D').sum(min_count=1),
        ds[MEAN_VARS].resample(time='1D').mean(),
    ])
    ds_daily = add_snow_rain(ds_daily)
    print(f'    Loading {year} into memory...')
    return ds_daily.load()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain',    required=True)
    ap.add_argument('--wy_start',  type=int, default=2015)
    ap.add_argument('--wy_end',    type=int, default=2021)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    out_zarr = domain_dir(args.domain) / 'aorc.zarr'
    if out_zarr.exists() and not args.overwrite:
        print('aorc.zarr already exists. Use --overwrite to redo.')
        return

    hrus = load_hrus(args.domain)
    bbox = domain_bbox(hrus)

    print('Connecting to AORC S3...')
    s3 = s3fs.S3FileSystem(anon=True)

    S       = None
    hru_ids = None
    accum   = {v: [] for v in OUT_VARS}
    times   = []

    for wy in range(args.wy_start, args.wy_end + 1):
        wy_start = pd.Timestamp(wy - 1, 10, 1)
        wy_end   = pd.Timestamp(wy, 9, 30, 23, 59, 59)
        print(f'  WY{wy} ({wy_start.date()} to {wy_end.date()})...')

        # fetch both calendar years for this WY in parallel
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = {yr: ex.submit(load_aorc_year, yr, bbox, s3)
                    for yr in [wy - 1, wy]}
        yearly = []
        for yr in [wy - 1, wy]:
            try:
                yearly.append(futs[yr].result())
            except Exception as e:
                print(f'    Warning: could not load {yr}: {e}')
        if not yearly:
            print(f'    No data for WY{wy}, skipping')
            continue

        ds_wy = xr.concat(yearly, dim='time').sel(time=slice(wy_start, wy_end))

        if S is None:
            print('  Rasterizing HRUs on AORC grid...')
            mask, hru_ids = rasterize_hrus(
                hrus, ds_wy.latitude.values, ds_wy.longitude.values
            )
            n_hrus = len(hru_ids)
            S = build_weight_matrix(mask, n_hrus)
            print(f'  {n_hrus} HRUs, {S.nnz} rasterized pixels')

        times.append(ds_wy.time.values)
        for var in OUT_VARS:
            data = ds_wy[var].values.astype(np.float32)
            accum[var].append(zonal_means(data, S))

    if not times:
        print('No AORC data found for the requested WY range.')
        return

    time_arr = np.concatenate(times)
    data_vars = {
        v: xr.DataArray(np.concatenate(accum[v], axis=0), dims=['time', 'divide_id'])
        for v in OUT_VARS
    }
    out_ds = xr.Dataset(data_vars, coords={'time': time_arr, 'divide_id': hru_ids})

    if out_zarr.exists():
        shutil.rmtree(out_zarr)
    out_ds.to_zarr(out_zarr, zarr_format=2)
    print(f'Saved aorc.zarr ({len(time_arr)} days x {len(hru_ids)} HRUs) → {out_zarr}')


if __name__ == '__main__':
    main()
