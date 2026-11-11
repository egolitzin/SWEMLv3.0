"""
Extract GridMET daily forcings per HRU via area-weighted zonal means.

Streams directly from THREDDS OPeNDAP -- no local files required.
    URL: thredds.northwestknowledge.net:8080/thredds/dodsC/MET/{var}/{var}_{year}.nc
    ~4 km CONUS daily, EPSG:4326

Input variables: pr, tmmx, tmmn, rmax, rmin, sph, vs, srad
Derived:
    tmean        = (tmmx + tmmn) / 2
    rmean        = (rmax + rmin) / 2
    Tw           = wet-bulb temperature (K), Stull (2011) from daily-mean T + RH
    SNOW_wetbulb = pr * P_snow via Wang (2019) on Stull (2011) Tw from tmean + rmean
    RAIN_wetbulb = pr * (1 - P_snow)

Processed one month at a time to bound peak memory (~120 MB for WUS-scale domains).

Output: model_domain_{name}/gridmet.zarr  (time, divide_id)

Usage:
    python 04_extract_gridmet.py --domain aso
    python 04_extract_gridmet.py --domain aso --wy_start 2015 --wy_end 2021
"""
import argparse
import shutil
import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix
from domain_utils import domain_dir, load_hrus, domain_bbox

THREDDS_BASE = (
    'http://thredds.northwestknowledge.net:8080'
    '/thredds/dodsC/MET/{var}/{var}_{year}.nc'
)
GRIDMET_VARS = ['pr', 'tmmx', 'tmmn', 'rmax', 'rmin', 'sph', 'vs', 'srad']
OUT_VARS     = GRIDMET_VARS + ['tmean', 'rmean', 'Tw', 'SNOW_wetbulb', 'RAIN_wetbulb']


def precip_phase(pr, tmmx_K, tmmn_K, rmax_pct, rmin_pct):
    """
    Phase-partition precip using Wang (2019) sigmoid on Stull (2011) Tw.
    Tw computed from daily-mean T and daily-mean RH.
    Returns (snow, rain) float32 in same units as pr.
    """
    T_C = (tmmx_K + tmmn_K) / 2.0 - 273.15
    RH  = np.clip((rmax_pct + rmin_pct) / 2.0, 0.0, 100.0)
    Tw  = (T_C * np.arctan(0.151977 * (RH + 8.313659) ** 0.5)
           + np.arctan(T_C + RH)
           - np.arctan(RH - 1.676331)
           + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH)
           - 4.686035)
    P_snow = 1.0 / (1.0 + 6.99e-5 * np.exp(2.0 * (Tw + 3.97)))
    return (pr * P_snow).astype(np.float32), (pr * (1.0 - P_snow)).astype(np.float32), (Tw + 273.15).astype(np.float32)


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
        raise ValueError('no HRUs overlap the GridMET grid; check domain bbox')
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


def open_gridmet_wy(wy: int, bbox: tuple) -> xr.Dataset | None:
    """
    Open GridMET for one water year (Oct wy-1 to Sep wy) lazily from THREDDS.
    No data is loaded -- callers slice and load one month at a time.
    Returns None if any variable file is unavailable.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    parts = []
    for cy in (wy - 1, wy):
        var_arrays = {}
        for var in GRIDMET_VARS:
            url = THREDDS_BASE.format(var=var, year=cy)
            try:
                ds_v = xr.open_dataset(url, engine='netcdf4')
                # internal variable names differ from filename stems (e.g. 'precipitation_amount');
                # pick the one data variable that has lat/lon dimensions
                lat_dim = 'latitude' if 'latitude' in ds_v.dims else 'lat'
                spatial = [v for v in ds_v.data_vars if lat_dim in ds_v[v].dims]
                if not spatial:
                    raise KeyError(f'no spatial data variable found; got {list(ds_v.data_vars)}')
                da = ds_v[spatial[0]]
            except Exception as e:
                print(f'  Cannot open {var} CY{cy}: {e}')
                return None

            rename = {}
            if 'latitude' in da.dims:
                rename['latitude'] = 'lat'
            if 'longitude' in da.dims:
                rename['longitude'] = 'lon'
            if 'day' in da.dims:
                rename['day'] = 'time'
            if rename:
                da = da.rename(rename)

            lat_vals  = da.lat.values
            lat_slice = (slice(lat_min, lat_max) if lat_vals[0] < lat_vals[-1]
                         else slice(lat_max, lat_min))
            var_arrays[var] = da.sel(lat=lat_slice, lon=slice(lon_min, lon_max))

        parts.append(xr.Dataset(var_arrays))

    wy_start = pd.Timestamp(wy - 1, 10, 1)
    wy_end   = pd.Timestamp(wy, 9, 30, 23, 59, 59)
    return xr.concat(parts, dim='time').sel(time=slice(wy_start, wy_end))


def process_month(ds_m: xr.Dataset, lat_desc: bool, S: csr_matrix) -> dict:
    """
    Extract zonal means for one already-loaded monthly slice.
    Returns dict var -> (n_days, n_hrus) float32, plus '_times'.
    """
    def get(var):
        data = ds_m[var].values.astype(np.float32)
        return data[:, ::-1, :] if lat_desc else data

    pr   = get('pr')
    tmmx = get('tmmx')
    tmmn = get('tmmn')
    rmax = get('rmax')
    rmin = get('rmin')

    snow, rain, tw = precip_phase(pr, tmmx, tmmn, rmax, rmin)
    tmean = ((tmmx + tmmn) / 2.0).astype(np.float32)
    rmean = ((rmax + rmin) / 2.0).astype(np.float32)

    fetched = {'pr': pr, 'tmmx': tmmx, 'tmmn': tmmn, 'rmax': rmax, 'rmin': rmin}
    result  = {}
    for var in GRIDMET_VARS:
        arr = fetched[var] if var in fetched else get(var)
        result[var] = zonal_means(arr, S)
    result['tmean']        = zonal_means(tmean, S)
    result['rmean']        = zonal_means(rmean, S)
    result['Tw']           = zonal_means(tw,    S)
    result['SNOW_wetbulb'] = zonal_means(snow,  S)
    result['RAIN_wetbulb'] = zonal_means(rain,  S)
    result['_times']       = ds_m.time.values
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain',    required=True)
    ap.add_argument('--wy_start',  type=int, default=2015)
    ap.add_argument('--wy_end',    type=int, default=2021)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    out_zarr = domain_dir(args.domain) / 'gridmet.zarr'
    if out_zarr.exists() and not args.overwrite:
        print('gridmet.zarr already exists. Use --overwrite to redo.')
        return

    hrus = load_hrus(args.domain)
    bbox = domain_bbox(hrus)

    S           = None
    hru_ids     = None
    first_write = True

    for wy in range(args.wy_start, args.wy_end + 1):
        print(f'  WY{wy}...')
        ds = open_gridmet_wy(wy, bbox)
        if ds is None:
            print(f'    Skipping WY{wy}')
            continue

        if S is None:
            lat_vals = ds.lat.values
            lat_asc  = lat_vals if lat_vals[0] < lat_vals[-1] else lat_vals[::-1]
            print('  Building coverage weight matrix...')
            S, hru_ids = build_weight_matrix(hrus, lat_asc, ds.lon.values)

        lat_desc    = ds.lat.values[0] > ds.lat.values[-1]
        wy_start_ts = pd.Timestamp(wy - 1, 10, 1)
        wy_end_ts   = pd.Timestamp(wy, 9, 30, 23, 59, 59)

        wy_accum = {v: [] for v in OUT_VARS}
        wy_times = []

        for m_start in pd.date_range(wy_start_ts, wy_end_ts, freq='MS'):
            m_end = min(m_start + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23), wy_end_ts)
            try:
                ds_m = ds.sel(time=slice(m_start, m_end)).load()
            except Exception as e:
                print(f'    Warning: {m_start.strftime("%Y-%m")}: {e}')
                continue

            monthly = process_month(ds_m, lat_desc, S)
            ds_m.close()

            wy_times.append(monthly.pop('_times'))
            for v in OUT_VARS:
                wy_accum[v].append(monthly[v])

        if not wy_times:
            print(f'    No data for WY{wy}, skipping')
            ds.close()
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

        ds.close()
        print(f'    WY{wy}: {len(time_arr)} days written')

    if not first_write:
        print(f'Saved gridmet.zarr → {out_zarr}')
    else:
        print('No GridMET data found for the requested WY range.')


if __name__ == '__main__':
    main()
