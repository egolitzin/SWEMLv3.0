"""
Extract NLDAS-2 FORA daily forcings per HRU via area-weighted zonal means.

Streams hourly files via authenticated HTTPS (GES DISC) -- no local files.
    Product: NLDAS_FORA0125_H v2.0, 1/8 degree (~12.5 km), hourly, EPSG:4326
    Base URL: https://data.gesdisc.earthdata.nasa.gov/data/NLDAS/NLDAS_FORA0125_H.2.0/
    Requires Earthdata credentials in ~/.netrc (machine urs.earthdata.nasa.gov)
    Downloads are parallelized (8 threads) to bound wall time per month.

Input variables (hourly):
    Rainf    total precipitation (kg/m2 = mm)
    Tair     2-m air temperature (K)
    Qair     2-m specific humidity (kg/kg)
    PSurf    surface pressure (Pa)
    SWdown   shortwave radiation flux downwards (W/m2)
    LWdown   longwave radiation flux downwards (W/m2)
    Wind_E   10-m zonal wind speed (m/s)
    Wind_N   10-m meridional wind speed (m/s)

Daily aggregation: Rainf summed; Tair min/mean/max; all others mean.
Phase partitioning is applied hourly before daily aggregation.
Derived:
    SNOW_wetbulb = Rainf * P_snow via Wang (2019) on Stull (2011) Tw
    RAIN_wetbulb = Rainf * (1 - P_snow)

Output: model_domain_{name}/nldas.zarr  (time, divide_id)

Usage:
    python 04_extract_nldas.py --domain aso
    python 04_extract_nldas.py --domain aso --wy_start 2015 --wy_end 2021
"""
import argparse
import io
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr
import earthaccess
from scipy.sparse import csr_matrix
from domain_utils import domain_dir, load_hrus, domain_bbox

HTTPS_BASE  = 'https://data.gesdisc.earthdata.nasa.gov/data/NLDAS/NLDAS_FORA0125_H.2.0'
HOURLY_VARS = ['Rainf', 'Tair', 'Qair', 'PSurf', 'SWdown', 'LWdown', 'Wind_E', 'Wind_N']


def nldas_https_url(dt: pd.Timestamp) -> str:
    return (
        f'{HTTPS_BASE}/{dt.year}/{dt.day_of_year:03d}'
        f'/NLDAS_FORA0125_H.A{dt.strftime("%Y%m%d")}.{dt.hour:02d}00.020.nc'
    )


def _fetch(args) -> io.BytesIO | None:
    """Fetch one hourly file with exponential-backoff retry. Returns None on final failure."""
    session, url = args
    for attempt in range(5):
        try:
            r = session.get(url, timeout=120)
            r.raise_for_status()
            return io.BytesIO(r.content)
        except Exception:
            if attempt < 4:
                time.sleep(2 ** attempt)
    return None


def _nan_hour(template: xr.Dataset, dt: pd.Timestamp) -> xr.Dataset:
    """Single-timestep NaN Dataset with the same structure as template."""
    return xr.Dataset({
        v: xr.DataArray(
            np.full_like(template[v].values, np.nan),
            dims=template[v].dims,
            coords={d: template[d] for d in template[v].dims if d != 'time'} | {'time': [dt]},
        )
        for v in HOURLY_VARS
    })


def open_nldas_month(session, hours: pd.DatetimeIndex, n_workers: int = 8) -> tuple:
    """Download all hourly files for one month in parallel, return concatenated Dataset.

    Retries each file up to 5 times with exponential backoff. Hours that still fail
    are filled with NaN rather than dropping the entire month.
    Loads data before closing file handles -- xr.concat is lazy and the returned
    Dataset would be unbacked if we closed first.
    """
    urls = [nldas_https_url(dt) for dt in hours]
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        buffers = list(ex.map(_fetch, [(session, u) for u in urls]))

    n_failed = sum(b is None for b in buffers)
    if n_failed == len(buffers):
        del buffers
        raise RuntimeError('all hourly files failed after retries')
    if n_failed > 0:
        print(f'    {n_failed}/{len(buffers)} files failed after retries; filling with NaN')

    # open template from first successful buffer (opened once, reused for NaN placeholders)
    first_good = next(i for i, b in enumerate(buffers) if b is not None)
    template = xr.open_dataset(buffers[first_good], engine='h5netcdf')[HOURLY_VARS]

    datasets = [None] * len(buffers)
    for i, (buf, dt) in enumerate(zip(buffers, hours)):
        if i == first_good:
            datasets[i] = template
        elif buf is not None:
            datasets[i] = xr.open_dataset(buf, engine='h5netcdf')[HOURLY_VARS]
        else:
            datasets[i] = _nan_hour(template, dt)

    # compute per-day missing counts from download success/failure before releasing buffers
    missing_mask    = np.array([b is None for b in buffers], dtype=int)
    hours_series    = pd.Series(missing_mask, index=pd.DatetimeIndex(hours))
    missing_per_day = hours_series.resample('1D').sum().values.astype(int)
    total_per_day   = hours_series.resample('1D').count().values.astype(int)

    ds = xr.concat(datasets, dim='time').load()
    for d in datasets:
        d.close()
    del buffers
    return ds, missing_per_day, total_per_day


OUT_VARS = [
    'Rainf', 'Tair', 'Tair_min', 'Tair_max', 'Qair', 'PSurf',
    'SWdown', 'LWdown', 'Wind_E', 'Wind_N', 'SNOW_wetbulb', 'RAIN_wetbulb',
]


def precip_phase(rainf, tair_K, qair, psurf_Pa):
    """
    Phase-partition precip using Wang (2019) sigmoid on Stull (2011) Tw.
    Tw derived from hourly T, Qair, and PSurf (same path as AORC).
    Returns (snow, rain) float32 in mm.
    """
    T_C   = tair_K - 273.15
    P_hPa = psurf_Pa / 100.0
    es    = 6.112 * np.exp(17.67 * T_C / (T_C + 243.5))
    e     = qair * P_hPa / (0.622 + 0.378 * qair)
    RH    = np.clip(e / es * 100.0, 0.0, 100.0)
    Tw    = (T_C * np.arctan(0.151977 * (RH + 8.313659) ** 0.5)
             + np.arctan(T_C + RH)
             - np.arctan(RH - 1.676331)
             + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH)
             - 4.686035)
    P_snow = 1.0 / (1.0 + 6.99e-5 * np.exp(2.0 * (Tw + 3.97)))
    return (rainf * P_snow).astype(np.float32), (rainf * (1.0 - P_snow)).astype(np.float32)


def build_weight_matrix(hrus, lat: np.ndarray, lon: np.ndarray):
    """
    Build sparse (n_pixels, n_hrus) coverage-fraction weight matrix via exactextract.
    lat must be ascending (S-to-N). Returns S and hru_ids.
    """
    from exactextract import exact_extract
    import rioxarray  # noqa: F401

    hrus_4326 = hrus.to_crs('EPSG:4326').reset_index(drop=True)
    n_lat, n_lon = len(lat), len(lon)

    idx_raster = xr.DataArray(
        np.arange(n_lat * n_lon, dtype=np.float32).reshape(n_lat, n_lon),
        dims=['y', 'x'],
        coords={'y': lat, 'x': lon},
    ).rio.write_crs('EPSG:4326')

    result_df = exact_extract(
        idx_raster, hrus_4326, ['values', 'coverage'],
        include_cols=['divide_id'], output='pandas',
    )

    pix_list, hru_list, wt_list = [], [], []
    for hru_idx, (_, row) in enumerate(result_df.iterrows()):
        pix_idx  = np.round(np.asarray(row['values'])).astype(np.int64)
        coverage = np.asarray(row['coverage'], dtype=np.float64)
        if pix_idx.size == 0:
            continue
        pix_list.append(pix_idx)
        hru_list.append(np.full(len(pix_idx), hru_idx, dtype=np.int64))
        wt_list.append(coverage)

    if not pix_list:
        raise ValueError('no HRUs overlap the NLDAS grid; check domain bbox')
    S = csr_matrix(
        (np.concatenate(wt_list),
         (np.concatenate(pix_list), np.concatenate(hru_list))),
        shape=(n_lat * n_lon, len(hrus_4326)),
    )
    print(f'  {len(hrus_4326)} HRUs, {S.nnz} pixel-HRU pairs (coverage-weighted)')
    return S, hrus_4326['divide_id'].values


def zonal_means(data: np.ndarray, S: csr_matrix) -> np.ndarray:
    """Coverage-weighted mean. data: (n_time, n_lat, n_lon). Returns (n_time, n_hrus) float32."""
    flat       = data.reshape(len(data), -1).astype(np.float64)
    valid_data = np.where(np.isfinite(flat), flat, 0.0)
    count_data = np.isfinite(flat).astype(np.float64)
    sums       = S.T.dot(valid_data.T).T
    counts     = S.T.dot(count_data.T).T
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(counts > 0, sums / counts, np.nan).astype(np.float32)


def aggregate_daily(ds_hourly: xr.Dataset, lat_desc: bool, S: csr_matrix,
                    missing_per_day: np.ndarray, total_per_day: np.ndarray) -> dict:
    """
    Resample hourly NLDAS to daily and extract zonal means.
    Phase partitioning is applied hourly before daily aggregation.
    missing_per_day / total_per_day come from open_nldas_month (download results,
    not data values) so they are immune to spatial masking artifacts.
    Returns dict var -> (n_days, n_hrus) float32.
    """
    def flip(arr):
        return arr[:, ::-1, :] if lat_desc else arr

    # hourly phase partitioning before aggregating to daily
    rainf_hr = ds_hourly['Rainf'].values.astype(np.float32)
    tair_hr  = ds_hourly['Tair'].values.astype(np.float32)
    qair_hr  = ds_hourly['Qair'].values.astype(np.float32)
    psurf_hr = ds_hourly['PSurf'].values.astype(np.float32)
    snow_hr, rain_hr = precip_phase(rainf_hr, tair_hr, qair_hr, psurf_hr)

    ds_ext = ds_hourly.assign({
        'SNOW_wetbulb': xr.DataArray(snow_hr, dims=ds_hourly['Rainf'].dims,
                                     coords=ds_hourly['Rainf'].coords),
        'RAIN_wetbulb': xr.DataArray(rain_hr, dims=ds_hourly['Rainf'].dims,
                                     coords=ds_hourly['Rainf'].coords),
    })

    if missing_per_day.max() > 2:
        n_bad = int((missing_per_day > 2).sum())
        print(f'    Warning: {n_bad} days have >2 missing hours (max {missing_per_day.max()}); '
              f'daily sums rescaled proportionally')
    valid_per_day = (total_per_day - missing_per_day).astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        scale = np.where(valid_per_day > 0,
                         total_per_day.astype(np.float64) / valid_per_day,
                         1.0).astype(np.float32)

    daily = xr.merge([
        ds_ext[['Rainf', 'SNOW_wetbulb', 'RAIN_wetbulb']].resample(time='1D').sum(min_count=1),
        ds_hourly[['Qair', 'PSurf', 'SWdown', 'LWdown', 'Wind_E', 'Wind_N']].resample(time='1D').mean(),
        ds_hourly[['Tair']].resample(time='1D').mean(),
        ds_hourly[['Tair']].resample(time='1D').min().rename({'Tair': 'Tair_min'}),
        ds_hourly[['Tair']].resample(time='1D').max().rename({'Tair': 'Tair_max'}),
    ]).load()
    ds_ext.close()  # release snow_hr/rain_hr arrays (~300 MB) before building result

    if len(scale) != len(daily.time):
        raise ValueError(
            f'scale length ({len(scale)}) != daily time steps ({len(daily.time)}); '
            f'file timestamps do not align with expected hourly schedule'
        )

    result = {
        'Rainf':        zonal_means(flip(daily['Rainf'].values.astype(np.float32)),        S) * scale[:, np.newaxis],
        'Tair':         zonal_means(flip(daily['Tair'].values.astype(np.float32)),          S),
        'Tair_min':     zonal_means(flip(daily['Tair_min'].values.astype(np.float32)),      S),
        'Tair_max':     zonal_means(flip(daily['Tair_max'].values.astype(np.float32)),      S),
        'Qair':         zonal_means(flip(daily['Qair'].values.astype(np.float32)),          S),
        'PSurf':        zonal_means(flip(daily['PSurf'].values.astype(np.float32)),         S),
        'SWdown':       zonal_means(flip(daily['SWdown'].values.astype(np.float32)),        S),
        'LWdown':       zonal_means(flip(daily['LWdown'].values.astype(np.float32)),        S),
        'Wind_E':       zonal_means(flip(daily['Wind_E'].values.astype(np.float32)),        S),
        'Wind_N':       zonal_means(flip(daily['Wind_N'].values.astype(np.float32)),        S),
        'SNOW_wetbulb': zonal_means(flip(daily['SNOW_wetbulb'].values.astype(np.float32)), S) * scale[:, np.newaxis],
        'RAIN_wetbulb': zonal_means(flip(daily['RAIN_wetbulb'].values.astype(np.float32)), S) * scale[:, np.newaxis],
    }
    result['_times'] = daily.time.values
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain',    required=True)
    ap.add_argument('--wy_start',  type=int, default=2015)
    ap.add_argument('--wy_end',    type=int, default=2021)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    # scale download threads to SLURM allocation; fall back to 8 outside SLURM
    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
    print(f'  Download threads: {n_workers} (SLURM_CPUS_PER_TASK={os.environ.get("SLURM_CPUS_PER_TASK", "unset")})')

    out_zarr = domain_dir(args.domain) / 'nldas.zarr'
    if out_zarr.exists() and not args.overwrite:
        print('nldas.zarr already exists. Use --overwrite to redo.')
        return

    hrus = load_hrus(args.domain)
    bbox = domain_bbox(hrus)
    lon_min, lat_min, lon_max, lat_max = bbox

    earthaccess.login(strategy='netrc')

    S           = None
    hru_ids     = None
    first_write = True

    for wy in range(args.wy_start, args.wy_end + 1):
        wy_start = pd.Timestamp(wy - 1, 10, 1)
        wy_end   = pd.Timestamp(wy, 9, 30, 23, 59, 59)
        print(f'  WY{wy} ({wy_start.date()} to {wy_end.date()})...')

        wy_accum = {v: [] for v in OUT_VARS}
        wy_times = []

        # refresh session each WY in case the token expires between months
        session = earthaccess.get_requests_https_session()

        month_starts = pd.date_range(wy_start, wy_end, freq='MS')
        for m_start in month_starts:
            m_end = m_start + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23)
            m_end = min(m_end, wy_end)

            hours = pd.date_range(m_start, m_end, freq='h')
            print(f'    {m_start.strftime("%Y-%m")}: downloading {len(hours)} hourly files...')
            try:
                ds_hr, missing_per_day, total_per_day = open_nldas_month(
                    session, hours, n_workers=n_workers)
            except Exception as e:
                print(f'    Warning: could not open {m_start.strftime("%Y-%m")}: {e}')
                continue

            # normalize coordinate names
            rename = {}
            if 'latitude' in ds_hr.dims:
                rename['latitude'] = 'lat'
            if 'longitude' in ds_hr.dims:
                rename['longitude'] = 'lon'
            if rename:
                ds_hr = ds_hr.rename(rename)

            # clip to domain bbox -- HTTPS files cover full CONUS
            lat_vals_raw = ds_hr.lat.values
            lat_slice = (slice(lat_min, lat_max) if lat_vals_raw[0] < lat_vals_raw[-1]
                         else slice(lat_max, lat_min))
            ds_hr = ds_hr.sel(lat=lat_slice, lon=slice(lon_min, lon_max))
            if ds_hr.lat.size == 0 or ds_hr.lon.size == 0:
                print(f'    Warning: bbox clip returned empty grid for {m_start.strftime("%Y-%m")}, skipping')
                ds_hr.close()
                continue

            if S is None:
                lat_vals = ds_hr.lat.values
                lat_asc  = lat_vals if lat_vals[0] < lat_vals[-1] else lat_vals[::-1]
                print('  Building coverage weight matrix...')
                S, hru_ids = build_weight_matrix(hrus, lat_asc, ds_hr.lon.values)

            lat_desc = ds_hr.lat.values[0] > ds_hr.lat.values[-1]

            try:
                monthly = aggregate_daily(ds_hr, lat_desc, S, missing_per_day, total_per_day)
            except Exception as e:
                print(f'    Warning: aggregation failed {m_start.strftime("%Y-%m")}: {e}')
                ds_hr.close()
                continue

            wy_times.append(monthly.pop('_times'))
            for v in OUT_VARS:
                wy_accum[v].append(monthly[v])
            ds_hr.close()

        if not wy_times:
            print(f'    No data for WY{wy}, skipping')
            continue

        time_arr  = np.concatenate(wy_times)
        data_vars = {
            v: xr.DataArray(np.concatenate(wy_accum[v], axis=0), dims=['time', 'divide_id'])
            for v in OUT_VARS
        }
        wy_ds = xr.Dataset(data_vars, coords={'time': time_arr, 'divide_id': hru_ids})

        if first_write:
            if out_zarr.exists():
                shutil.rmtree(out_zarr)
            wy_ds.to_zarr(out_zarr, zarr_format=2)
            first_write = False
        else:
            wy_ds.to_zarr(out_zarr, append_dim='time')

        print(f'    WY{wy}: {len(time_arr)} days written')

    if not first_write:
        print(f'Saved nldas.zarr → {out_zarr}')
    else:
        print('No NLDAS data found for the requested WY range.')


if __name__ == '__main__':
    main()
