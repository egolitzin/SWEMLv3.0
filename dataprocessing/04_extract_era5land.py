"""
Extract ERA5-Land daily forcings per HRU via area-weighted zonal means.

Uses the CDS API (cdsapi) to request monthly data on the fly -- no persistent
local files. Each month is downloaded to a temp file, processed, and deleted.
Requires a CDS account and ~/.cdsapirc with your API key:
    https://cds.climate.copernicus.eu/how-to-api

Input variables (hourly):
    t2m    2-m temperature (K)
    d2m    2-m dewpoint temperature (K)
    tp     total precipitation (m, cumulative from 00:00 UTC)
    sf     snowfall (m water equiv, cumulative from 00:00 UTC)
    sp     surface pressure (Pa)
    # u10    10-m zonal wind (m/s)
    # v10    10-m meridional wind (m/s)
    ssrd   surface solar radiation downwards (J/m2, cumulative from 00:00 UTC)
    # strd   surface thermal radiation downwards (J/m2, cumulative from 00:00 UTC)

Daily aggregation:
    tp, sf   summed and converted to mm/day
    ssrd, strd  summed J/m2/day
    t2m      mean, min, max; d2m mean
    sp, u10, v10  mean

Derived:
    sf_frac      = daily sf / tp (ERA5-Land model phase diagnostic, analogous to HRRR cpofp)
    Tw           = wet-bulb temperature (K), Stull (2011) from hourly t2m + d2m; Bolton (1980) RH
    SNOW_wetbulb = tp * P_snow via Wang (2019) on Stull (2011) Tw
    RAIN_wetbulb = tp * (1 - P_snow)

Output: model_domain_{name}/era5land.zarr  (time, divide_id)

Usage:
    python 04_extract_era5land.py --domain aso
    python 04_extract_era5land.py --domain aso --wy_start 2015 --wy_end 2021
    python 04_extract_era5land.py --domain aso --resume --wy_end 2025
    python 04_extract_era5land.py --domain aso --resume_from 2020 --wy_end 2025
"""
import argparse
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix
from domain_utils import domain_dir, load_hrus, domain_bbox

# Grouped to 3 variables per CDS request (1 var/request = 9 requests/month).
CDS_VAR_GROUPS = [
    ['2m_temperature', '2m_dewpoint_temperature', 'surface_pressure'],
    ['total_precipitation', 'snowfall', 'surface_solar_radiation_downwards'],
    # ['surface_thermal_radiation_downwards',
    #  '10m_u_component_of_wind', '10m_v_component_of_wind'],
]
OUT_VARS = [
    'tp', 'sf_frac', 'SNOW_wetbulb', 'RAIN_wetbulb',
    't2m', 't2m_min', 't2m_max', 'd2m', 'Tw',
    'sp', 
    # 'u10', 'v10', 
    'ssrd', 
    # 'strd',
]


def precip_phase(tp_mm, t2m_K, d2m_K):
    """
    Phase-partition precip using Wang (2019) sigmoid on Stull (2011) Tw.
    RH derived from t2m/d2m via Bolton (1980) saturation vapor pressure.
    Returns (snow, rain) float32 in mm.
    """
    T_C  = t2m_K - 273.15
    Td_C = d2m_K - 273.15
    es   = 6.112 * np.exp(17.67 * T_C  / (T_C  + 243.5))
    ed   = 6.112 * np.exp(17.67 * Td_C / (Td_C + 243.5))
    RH   = np.clip(ed / es * 100.0, 0.0, 100.0)
    Tw = (T_C * np.arctan(0.151977 * (RH + 8.313659) ** 0.5)
          + np.arctan(T_C + RH)
          - np.arctan(RH - 1.676331)
          + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH)
          - 4.686035)
    P_snow = 1.0 / (1.0 + 6.99e-5 * np.exp(2.0 * (Tw + 3.97)))
    return (tp_mm * P_snow).astype(np.float32), (tp_mm * (1.0 - P_snow)).astype(np.float32)


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
        raise ValueError('no HRUs overlap the ERA5-Land grid; check domain bbox')
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


def fetch_era5land_month(c, year: int, month: int, bbox: tuple,
                         n_workers: int = len(CDS_VAR_GROUPS)) -> xr.Dataset:
    """
    Download one month of hourly ERA5-Land. The 3 var groups are submitted to
    CDS simultaneously (one thread per group, each with its own client) so queue
    wait time is paid once rather than three times sequentially.
    Field count per request: 3 vars x 31 days x 24 h = 2,232 (within CDS limit).
    Returns an in-memory Dataset.
    """
    import cdsapi as _cdsapi
    lon_min, lat_min, lon_max, lat_max = bbox
    base_req = {
        'product_type':    'reanalysis',
        'year':            str(year),
        'month':           f'{month:02d}',
        'day':             [f'{d:02d}' for d in range(1, 32)],
        'time':            [f'{h:02d}:00' for h in range(24)],
        'area':            [lat_max, lon_min, lat_min, lon_max],
        'data_format':     'netcdf',
        'download_format': 'unarchived',
    }
    cds_url, cds_key = c.url, c.key  # capture before entering threads

    tmp_paths = []
    for _ in CDS_VAR_GROUPS:
        tmp = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
        tmp.close()
        tmp_paths.append(tmp.name)

    def _download(group_path):
        group, path = group_path
        cl = _cdsapi.Client(url=cds_url, key=cds_key, quiet=True)
        cl.retrieve('reanalysis-era5-land', {**base_req, 'variable': group}, path)

    parts = []
    try:
        with ThreadPoolExecutor(max_workers=min(len(CDS_VAR_GROUPS), n_workers)) as ex:
            list(ex.map(_download, zip(CDS_VAR_GROUPS, tmp_paths)))
        parts = [xr.open_dataset(p, engine='netcdf4') for p in tmp_paths]
        merged = xr.merge(parts).load()
        if 'valid_time' in merged.dims:
            merged = merged.rename({'valid_time': 'time'})
        return merged
    finally:
        for ds in parts:
            ds.close()
        for p in tmp_paths:
            if os.path.exists(p):
                os.unlink(p)


def aggregate_daily(ds_hourly: xr.Dataset, lat_desc: bool, S: csr_matrix) -> dict:
    """
    Aggregate hourly ERA5-Land to daily and extract zonal means.

    tp, sf, ssrd, strd are cumulative from 00:00 UTC. diff() converts them to
    per-hour increments (midnight resets appear as negatives, clipped to 0).
    Phase partitioning is applied per hour before summing to daily.
    """
    # deaccumulate: cumulative → per-hour; diff labels result with the later timestamp
    tp_hr   = ds_hourly['tp'].diff(dim='time').clip(min=0)
    sf_hr   = ds_hourly['sf'].diff(dim='time').clip(min=0)
    ssrd_hr = ds_hourly['ssrd'].diff(dim='time').clip(min=0)
    # strd_hr = ds_hourly['strd'].diff(dim='time').clip(min=0)

    # phase partition at hourly resolution using instantaneous t2m/d2m
    t2m_hr = ds_hourly['t2m'].sel(time=tp_hr.time)
    d2m_hr = ds_hourly['d2m'].sel(time=tp_hr.time)
    snow_hr, rain_hr = precip_phase(tp_hr * 1000.0, t2m_hr, d2m_hr)

    # Tw from all 24h t2m/d2m via Bolton (1980) + Stull (2011)
    T_C_all  = ds_hourly['t2m'] - 273.15
    Td_C_all = ds_hourly['d2m'] - 273.15
    es_all   = 6.112 * np.exp(17.67 * T_C_all  / (T_C_all  + 243.5))
    ed_all   = 6.112 * np.exp(17.67 * Td_C_all / (Td_C_all + 243.5))
    RH_all   = np.clip(ed_all / es_all * 100.0, 0.0, 100.0)
    tw_all   = (T_C_all * np.arctan(0.151977 * (RH_all + 8.313659) ** 0.5)
                + np.arctan(T_C_all + RH_all)
                - np.arctan(RH_all - 1.676331)
                + 0.00391838 * RH_all ** 1.5 * np.arctan(0.023101 * RH_all)
                - 4.686035 + 273.15)  # K

    # daily aggregation
    daily = xr.merge([
        xr.Dataset({'tp': tp_hr, 'sf': sf_hr, 
                    'ssrd': ssrd_hr, 
                    # 'strd': strd_hr,
                    'SNOW_wetbulb': snow_hr, 'RAIN_wetbulb': rain_hr,
                    }).resample(time='1D').sum(min_count=1),
        ds_hourly[['t2m']].resample(time='1D').mean(),
        ds_hourly[['t2m']].resample(time='1D').min().rename({'t2m': 't2m_min'}),
        ds_hourly[['t2m']].resample(time='1D').max().rename({'t2m': 't2m_max'}),
        ds_hourly[['d2m']].resample(time='1D').mean(),
        ds_hourly[['sp']].resample(time='1D').mean(),  #, 'u10', 'v10'
        xr.Dataset({'Tw': tw_all}).resample(time='1D').mean(),
    ]).load()

    def flip(arr):
        return arr[:, ::-1, :] if lat_desc else arr

    # tp and sf need m → mm; SNOW/RAIN_wetbulb already in mm (precip_phase * 1000)
    tp   = flip(daily['tp'].values.astype(np.float32))   * 1000.0
    sf   = flip(daily['sf'].values.astype(np.float32))   * 1000.0
    snow = flip(daily['SNOW_wetbulb'].values.astype(np.float32))
    rain = flip(daily['RAIN_wetbulb'].values.astype(np.float32))

    with np.errstate(invalid='ignore', divide='ignore'):
        sf_frac = np.where(tp > 0, np.clip(sf / tp, 0.0, 1.0), np.nan).astype(np.float32)

    result = {
        'tp':           zonal_means(tp,      S),
        'sf_frac':      zonal_means(sf_frac, S),
        'SNOW_wetbulb': zonal_means(snow,    S),
        'RAIN_wetbulb': zonal_means(rain,    S),
        't2m':          zonal_means(flip(daily['t2m'].values.astype(np.float32)),     S),
        't2m_min':      zonal_means(flip(daily['t2m_min'].values.astype(np.float32)), S),
        't2m_max':      zonal_means(flip(daily['t2m_max'].values.astype(np.float32)), S),
        'd2m':          zonal_means(flip(daily['d2m'].values.astype(np.float32)),     S),
        'Tw':           zonal_means(flip(daily['Tw'].values.astype(np.float32)),      S),
        'sp':           zonal_means(flip(daily['sp'].values.astype(np.float32)),      S),
        # 'u10':          zonal_means(flip(daily['u10'].values.astype(np.float32)),     S),
        # 'v10':          zonal_means(flip(daily['v10'].values.astype(np.float32)),     S),
        'ssrd':         zonal_means(flip(daily['ssrd'].values.astype(np.float32)),    S),
        # 'strd':         zonal_means(flip(daily['strd'].values.astype(np.float32)),    S),
        '_times':       daily.time.values,
    }
    return result


def _truncate_zarr_before_wy(out_zarr, wy: int) -> bool:
    """Rewrite out_zarr keeping only timesteps before the start of WY wy.
    Returns True if any data remains; False if the zarr was removed entirely."""
    cutoff = np.datetime64(pd.Timestamp(wy - 1, 10, 1))
    print(f'Truncating era5land.zarr: keeping times before {pd.Timestamp(cutoff).date()}...')
    ds = xr.open_zarr(out_zarr, consolidated=False).compute()
    trimmed = ds.sel(time=ds.time < cutoff)
    if len(trimmed.time) == 0:
        print('  No data before that WY; will create fresh zarr.')
        shutil.rmtree(out_zarr)
        return False
    tmp = out_zarr.parent / '_era5land_truncate_tmp.zarr'
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
                    help='delete existing era5land.zarr and reprocess all WYs')
    ap.add_argument('--resume',      action='store_true',
                    help='auto-detect last complete WY in zarr and append from the next one')
    ap.add_argument('--resume_from', type=int, metavar='WY',
                    help='truncate zarr to before WY start, then reprocess from that WY')
    ap.add_argument('--cds_key',     default=None,
                    help='CDS API token; if omitted reads from ~/.cdsapirc')
    args = ap.parse_args()

    out_zarr    = domain_dir(args.domain) / 'era5land.zarr'
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
            print('era5land.zarr already exists. Use --overwrite, --resume, or --resume_from WY.')
            return

    # use SLURM allocation when available; cap at number of var groups (always 3)
    n_workers = min(len(CDS_VAR_GROUPS),
                    int(os.environ.get('SLURM_CPUS_PER_TASK', len(CDS_VAR_GROUPS))))
    print(f'  CDS download threads: {n_workers}')

    hrus = load_hrus(args.domain)
    bbox = domain_bbox(hrus)

    try:
        import cdsapi
    except ImportError:
        raise ImportError('cdsapi not installed. Run: pip install cdsapi')
    c = (cdsapi.Client(url='https://cds.climate.copernicus.eu/api', key=args.cds_key)
         if args.cds_key else cdsapi.Client())

    S       = None
    hru_ids = None

    for wy in range(args.wy_start, args.wy_end + 1):
        wy_start = pd.Timestamp(wy - 1, 10, 1)
        wy_end   = pd.Timestamp(wy, 9, 30, 23, 59, 59)
        print(f'  WY{wy} ({wy_start.date()} to {wy_end.date()})...')

        wy_accum = {v: [] for v in OUT_VARS}
        wy_times = []

        for m_start in pd.date_range(wy_start, wy_end, freq='MS'):
            year  = m_start.year
            month = m_start.month
            print(f'    {m_start.strftime("%Y-%m")}...', flush=True)

            try:
                ds_hr = fetch_era5land_month(c, year, month, bbox, n_workers=n_workers)
            except Exception as e:
                print(f'    Warning: CDS request failed {m_start.strftime("%Y-%m")}: {e}')
                continue

            try:
                ds_hr = ds_hr.sel(time=slice(wy_start, wy_end))

                if S is None:
                    lat_vals = ds_hr.latitude.values
                    lat_asc  = lat_vals if lat_vals[0] < lat_vals[-1] else lat_vals[::-1]
                    print('  Building coverage weight matrix...')
                    S, hru_ids = build_weight_matrix(hrus, lat_asc, ds_hr.longitude.values)

                lat_desc = ds_hr.latitude.values[0] > ds_hr.latitude.values[-1]
                ds_hr    = ds_hr.rename({'latitude': 'lat', 'longitude': 'lon'})
                monthly  = aggregate_daily(ds_hr, lat_desc, S)
            except Exception as e:
                print(f'    Warning: processing failed {m_start.strftime("%Y-%m")}: {e}')
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
            wy_ds.to_zarr(out_zarr, zarr_format=2)
            first_write = False
        else:
            wy_ds.to_zarr(out_zarr, append_dim='time')

        print(f'    WY{wy}: {len(time_arr)} days written')

    if not first_write:
        print(f'Saved era5land.zarr → {out_zarr}')
    else:
        print('No ERA5-Land data found for the requested WY range.')


if __name__ == '__main__':
    main()
