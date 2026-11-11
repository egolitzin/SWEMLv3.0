"""
Extract AORC daily forcings per HRU via pixel-weighted zonal means.

Data source: s3://noaa-nws-aorc-v1-1-1km/{year}.zarr (public, no credentials)
    dims: (time, latitude, longitude), EPSG:4326, 1-km resolution
    variables: APCP_surface, TMP_2maboveground, SPFH_2maboveground, PRES_surface,
               DSWRF_surface, DLWRF_surface, UGRD_10maboveground, VGRD_10maboveground

Derived variables added before extraction:
    Tw           - wet-bulb temperature (K), Stull (2011) from hourly T + RH
    SNOW_wetbulb - snow phase fraction via Wang (2019) wet-bulb sigmoid
    RAIN_wetbulb - rain phase fraction

Output: model_domain_{name}/aorc.zarr  (time, divide_id)

Each WY spans two calendar years. Years are opened lazily and processed one
month at a time (rechunked to 24 h) to bound peak memory. Each WY is written
to zarr immediately via append_dim so accumulation never exceeds one WY.

Usage:
    python 04_extract_aorc.py --domain sierra_nevada
    python 04_extract_aorc.py --domain sierra_nevada --wy_start 2015 --wy_end 2021
    python 04_extract_aorc.py --domain aso --resume --wy_end 2025
    python 04_extract_aorc.py --domain aso --resume_from 2020 --wy_end 2025
"""
import argparse
import shutil
import numpy as np
import pandas as pd
import xarray as xr
import s3fs
from scipy.sparse import csr_matrix
from domain_utils import domain_dir, load_hrus, domain_bbox

AORC_S3_BUCKET = 'noaa-nws-aorc-v1-1-1km'

MEAN_VARS = [
    'TMP_2maboveground', 'SPFH_2maboveground', 'PRES_surface',
    'DSWRF_surface', 'DLWRF_surface',
    'UGRD_10maboveground', 'VGRD_10maboveground',
]
SUM_VARS    = ['APCP_surface']
MINMAX_VARS = ['TMP_2maboveground']
OUT_VARS = (SUM_VARS + MEAN_VARS
            + [f'{v}_min' for v in MINMAX_VARS]
            + [f'{v}_max' for v in MINMAX_VARS]
            + ['Tw', 'SNOW_wetbulb', 'RAIN_wetbulb'])


def add_snow_rain(ds: xr.Dataset) -> xr.Dataset:
    """Add d2m, SNOW_wetbulb, and RAIN_wetbulb as derived hourly variables."""
    T_K   = ds['TMP_2maboveground']
    P_Pa  = ds['PRES_surface']
    P_hPa = P_Pa / 100.0
    q     = ds['SPFH_2maboveground']
    T_C   = T_K - 273.15

    # wet-bulb temperature via Stull (2011); output in K
    es = 6.112 * np.exp(17.67 * T_C / (T_C + 243.5))
    e  = q * P_hPa / (0.622 + 0.378 * q)
    RH = (e / es * 100.0).clip(0.0, 100.0)

    Tw = (T_C * np.arctan(0.151977 * (RH + 8.313659) ** 0.5)
          + np.arctan(T_C + RH)
          - np.arctan(RH - 1.676331)
          + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH)
          - 4.686035)
    ds['Tw'] = Tw + 273.15

    a = 6.99e-5
    b = 2.0
    c = 3.97
    P_snow = 1.0 / (1.0 + a * np.exp(b * (Tw + c)))

    ds['SNOW_wetbulb'] = ds['APCP_surface'] * P_snow
    ds['RAIN_wetbulb'] = ds['APCP_surface'] * (1.0 - P_snow)
    return ds


def open_aorc_year(year: int, bbox: tuple, s3: s3fs.S3FileSystem) -> xr.Dataset:
    """Open one calendar year of AORC from S3 lazily, clipped to bbox."""
    lon_min, lat_min, lon_max, lat_max = bbox
    store = s3fs.S3Map(root=f'{AORC_S3_BUCKET}/{year}.zarr', s3=s3, check=False)
    ds = xr.open_zarr(store, consolidated=True, chunks={})
    return ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))


def daily_from_hourly(ds: xr.Dataset) -> xr.Dataset:
    """Aggregate hourly AORC (with snow/rain added) to daily. Returns lazy dataset."""
    minmax = xr.merge([
        ds[MINMAX_VARS].resample(time='1D').min().rename(
            {v: f'{v}_min' for v in MINMAX_VARS}),
        ds[MINMAX_VARS].resample(time='1D').max().rename(
            {v: f'{v}_max' for v in MINMAX_VARS}),
    ])
    return xr.merge([
        ds[SUM_VARS + ['SNOW_wetbulb', 'RAIN_wetbulb']].resample(time='1D').sum(min_count=1),
        ds[MEAN_VARS + ['Tw']].resample(time='1D').mean(),
        minmax,
    ])


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


def _truncate_zarr_before_wy(out_zarr, wy: int) -> bool:
    """Rewrite out_zarr keeping only timesteps before the start of WY wy.
    Returns True if any data remains; False if the zarr was removed entirely."""
    cutoff = np.datetime64(pd.Timestamp(wy - 1, 10, 1))
    print(f'Truncating aorc.zarr: keeping times before {pd.Timestamp(cutoff).date()}...')
    ds = xr.open_zarr(out_zarr, consolidated=False).compute()
    trimmed = ds.sel(time=ds.time < cutoff)
    if len(trimmed.time) == 0:
        print('  No data before that WY; will create fresh zarr.')
        shutil.rmtree(out_zarr)
        return False
    tmp = out_zarr.parent / '_aorc_truncate_tmp.zarr'
    if tmp.exists():
        shutil.rmtree(tmp)
    trimmed.to_zarr(tmp, zarr_format=2)
    shutil.rmtree(out_zarr)
    tmp.rename(out_zarr)
    print(f'  Kept {len(trimmed.time)} timesteps through {pd.Timestamp(trimmed.time.values[-1]).date()}')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain',      required=True)
    ap.add_argument('--wy_start',    type=int, default=2015)
    ap.add_argument('--wy_end',      type=int, default=2021)
    ap.add_argument('--overwrite',   action='store_true',
                    help='delete existing aorc.zarr and reprocess all WYs')
    ap.add_argument('--resume',      action='store_true',
                    help='auto-detect last complete WY in zarr and append from the next one')
    ap.add_argument('--resume_from', type=int, metavar='WY',
                    help='truncate zarr to before WY start, then reprocess from that WY')
    args = ap.parse_args()

    out_zarr = domain_dir(args.domain) / 'aorc.zarr'
    first_write = True

    if out_zarr.exists():
        if args.overwrite:
            shutil.rmtree(out_zarr)
        elif args.resume_from:
            has_data = _truncate_zarr_before_wy(out_zarr, args.resume_from)
            args.wy_start = args.resume_from
            first_write = not has_data
        elif args.resume:
            ds_existing = xr.open_zarr(out_zarr, consolidated=False)
            last_ts = pd.Timestamp(ds_existing.time.values[-1])
            ds_existing.close()
            last_wy = last_ts.year + 1 if last_ts.month >= 10 else last_ts.year
            print(f'Resuming: last complete WY in zarr is WY{last_wy}, starting from WY{last_wy + 1}')
            args.wy_start = last_wy + 1
            first_write = False
        else:
            print('aorc.zarr already exists. Use --overwrite, --resume, or --resume_from WY.')
            return

    hrus = load_hrus(args.domain)
    bbox = domain_bbox(hrus)

    print('Connecting to AORC S3...')
    s3 = s3fs.S3FileSystem(anon=True)

    S       = None
    hru_ids = None

    for wy in range(args.wy_start, args.wy_end + 1):
        wy_start_ts = pd.Timestamp(wy - 1, 10, 1)
        wy_end_ts   = pd.Timestamp(wy, 9, 30, 23, 59, 59)
        print(f'  WY{wy} ({wy_start_ts.date()} to {wy_end_ts.date()})...')

        try:
            ds_prev = open_aorc_year(wy - 1, bbox, s3)
            ds_curr = open_aorc_year(wy,     bbox, s3)
        except Exception as e:
            print(f'    Warning: could not open AORC for WY{wy}: {e}')
            continue

        if S is None:
            print('  Rasterizing HRUs on AORC grid...')
            mask, hru_ids = rasterize_hrus(
                hrus, ds_prev.latitude.values, ds_prev.longitude.values)
            n_hrus = len(hru_ids)
            S = build_weight_matrix(mask, n_hrus)
            print(f'  {n_hrus} HRUs, {S.nnz} rasterized pixels')

        wy_times = []
        wy_accum = {v: [] for v in OUT_VARS}

        # Oct-Dec come from ds_prev (calendar year wy-1),
        # Jan-Sep come from ds_curr (calendar year wy).
        # Avoid xr.concat to prevent coordinate alignment errors between annual zarrs.
        # process one month at a time; rechunk to 24 h so dask materializes
        # one day of hourly data per task regardless of on-disk chunk size
        month_sources = (
            [(m, ds_prev) for m in pd.date_range(wy_start_ts,
                                                  pd.Timestamp(wy - 1, 12, 1), freq='MS')]
            + [(m, ds_curr) for m in pd.date_range(pd.Timestamp(wy, 1, 1),
                                                    wy_end_ts, freq='MS')]
        )
        for m_start, ds_src in month_sources:
            m_end = m_start + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23)
            try:
                ds_m  = ds_src.sel(time=slice(m_start, m_end)).chunk({'time': 24})
                ds_m  = add_snow_rain(ds_m)
                ds_daily = daily_from_hourly(ds_m).load()
            except Exception as e:
                print(f'    Warning: skipping {m_start.strftime("%Y-%m")}: {e}')
                continue

            wy_times.append(ds_daily.time.values)
            for var in OUT_VARS:
                data = ds_daily[var].values.astype(np.float32)
                wy_accum[var].append(zonal_means(data, S))
            del ds_daily

        if not wy_times:
            print(f'    No data for WY{wy}, skipping')
            continue

        time_arr = np.concatenate(wy_times)
        data_vars = {
            v: xr.DataArray(np.concatenate(wy_accum[v], axis=0), dims=['time', 'divide_id'])
            for v in OUT_VARS
        }
        wy_ds = xr.Dataset(data_vars, coords={'time': time_arr, 'divide_id': hru_ids})

        if first_write:
            wy_ds.to_zarr(out_zarr, zarr_format=2)
            first_write = False
        else:
            wy_ds.to_zarr(out_zarr, append_dim='time')

        print(f'    WY{wy}: {len(time_arr)} days written')

    if not first_write:
        print(f'Saved aorc.zarr → {out_zarr}')
    else:
        print('No AORC data found for the requested WY range.')


if __name__ == '__main__':
    main()
