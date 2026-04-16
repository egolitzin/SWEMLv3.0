"""
Extract HRRR daily forcings per HRU via area-weighted zonal means.

Data source: base_loc/HRRR/HRRR_daily_WY{WY}.nc
    dims: (time, y, x), 2D latitude/longitude coords (Lambert Conformal grid)
    variables: tp, t2m, sh2, d2m, t_sfc, sp, cpofp, frac_frozen,
               snow_wetbulb, rain_wetbulb, snow_cpofp, rain_cpofp

Output: model_domain_{name}/hrrr.zarr  (time, divide_id)

Usage:
    python 04_extract_hrrr.py --domain sierra_nevada
    python 04_extract_hrrr.py --domain sierra_nevada --wy_start 2015 --wy_end 2021
"""
import argparse
import shutil
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401
import geopandas as gpd
from pyproj import CRS, Transformer
from scipy.sparse import csr_matrix
from domain_utils import domain_dir, load_hrus, domain_bbox, BASE_LOC

HRRR_DIR  = BASE_LOC / 'HRRR'
HRRR_VARS = [
    'tp', 't2m', 'sh2', 'd2m', 't_sfc', 'sp', 'cpofp', 'frac_frozen',
    'snow_wetbulb', 'rain_wetbulb', 'snow_cpofp', 'rain_cpofp',
]

# HRRR native Lambert Conformal Conic projection (NCEP spherical earth)
HRRR_CRS = CRS.from_proj4(
    '+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 '
    '+R=6371229 +units=m +no_defs'
)


def latlon_to_lcc(lat2d: np.ndarray, lon2d: np.ndarray):
    """
    Convert HRRR 2D lat/lon to native LCC (x, y) and extract 1D coordinate
    arrays. HRRR is regular in LCC: x varies only with column, y only with row.
    Returns x_1d (nx,) and y_1d (ny,) in metres, both ascending.
    """
    tf = Transformer.from_crs('EPSG:4326', HRRR_CRS, always_xy=True)
    x2d, y2d = tf.transform(lon2d, lat2d)
    mid_row = x2d.shape[0] // 2
    mid_col = y2d.shape[1] // 2
    x_1d = x2d[mid_row, :]    # x values along middle row
    y_1d = y2d[:, mid_col]    # y values along middle column
    # ensure ascending y (S-to-N); x should already be ascending (W-to-E) for western US
    if y_1d[0] > y_1d[-1]:
        y_1d = y_1d[::-1]
    return x_1d, y_1d


def build_weight_matrix(hrus: gpd.GeoDataFrame,
                        x_1d: np.ndarray, y_1d: np.ndarray):
    """
    Build sparse (n_pixels, n_hrus) weight matrix S using exactextract coverage
    fractions. S[p, h] = fraction of pixel p covered by HRU h (0..1).
    Handles HRUs smaller than a HRRR pixel correctly: coverage is computed from
    actual polygon-pixel area overlap, not just pixel-center containment.
    x_1d, y_1d: 1D ascending pixel centers in LCC metres.
    Returns S and hru_ids array.
    """
    from exactextract import exact_extract

    n_y, n_x = len(y_1d), len(x_1d)
    hrus_lcc = hrus.to_crs(HRRR_CRS).reset_index(drop=True)

    # pixel values = flat S-to-N index so exact_extract returns which pixels
    # overlap each HRU (float32 exact for indices up to 2^24 ~ 16M)
    idx_raster = xr.DataArray(
        np.arange(n_y * n_x, dtype=np.float32).reshape(n_y, n_x),
        dims=['y', 'x'],
        coords={'y': y_1d, 'x': x_1d},
    ).rio.write_crs(HRRR_CRS)

    result_df = exact_extract(
        idx_raster, hrus_lcc, ['values', 'coverage'],
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
        raise ValueError('no HRUs overlap the clipped HRRR grid; check domain bbox and CRS')
    flat_pix = np.concatenate(pix_list)
    flat_hru = np.concatenate(hru_list)
    flat_wt  = np.concatenate(wt_list)
    S = csr_matrix((flat_wt, (flat_pix, flat_hru)),
                   shape=(n_y * n_x, len(hrus_lcc)))
    print(f'  {len(hrus_lcc)} HRUs, {S.nnz} pixel-HRU pairs (coverage-weighted)')
    return S, hrus_lcc['divide_id'].values


def zonal_means(data: np.ndarray, S: csr_matrix) -> np.ndarray:
    """
    Area-weighted mean per HRU for all time steps via sparse matrix multiply.
    data : (n_time, n_y, n_x), rows in S-to-N order (ascending y)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain',    required=True)
    ap.add_argument('--wy_start',  type=int, default=2015)
    ap.add_argument('--wy_end',    type=int, default=2021)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    out_zarr = domain_dir(args.domain) / 'hrrr.zarr'
    if out_zarr.exists() and not args.overwrite:
        print('hrrr.zarr already exists. Use --overwrite to redo.')
        return

    hrus = load_hrus(args.domain)
    bbox = domain_bbox(hrus)
    lon_min, lat_min, lon_max, lat_max = bbox

    S       = None
    hru_ids = None
    accum   = {v: [] for v in HRRR_VARS}
    times   = []

    for wy in range(args.wy_start, args.wy_end + 1):
        hrrr_path = HRRR_DIR / f'HRRR_daily_WY{wy}.nc'
        if not hrrr_path.exists():
            print(f'  WY{wy}: {hrrr_path} not found, skipping')
            continue

        print(f'  WY{wy}...')
        ds = xr.open_dataset(hrrr_path)

        lat2d = ds['latitude'].values
        lon2d = ds['longitude'].values
        # HRRR GRIB files store longitude in [0, 360]; convert to [-180, 180]
        if lon2d.max() > 180:
            lon2d = np.where(lon2d > 180, lon2d - 360, lon2d)

        in_box = (
            (lat2d >= lat_min) & (lat2d <= lat_max) &
            (lon2d >= lon_min) & (lon2d <= lon_max)
        )
        rows = np.where(in_box.any(axis=1))[0]
        cols = np.where(in_box.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            print(f'  WY{wy}: no HRRR pixels overlap domain bbox, skipping')
            ds.close()
            continue
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        lat2d_clip = lat2d[r0:r1, c0:c1]
        lon2d_clip = lon2d[r0:r1, c0:c1]
        ds = ds.isel(y=slice(r0, r1), x=slice(c0, c1))

        if S is None:
            print('  Converting HRRR grid to LCC and computing coverage weights...')
            x_1d, y_1d = latlon_to_lcc(lat2d_clip, lon2d_clip)
            S, hru_ids = build_weight_matrix(hrus, x_1d, y_1d)

        times.append(ds.time.values)
        for var in HRRR_VARS:
            if var not in ds:
                continue
            data = ds[var].values.astype(np.float32)  # (n_time, n_y, n_x)
            # HRRR NetCDF is stored N-to-S (row 0 = north); flip to S-to-N
            data = data[:, ::-1, :]
            accum[var].append(zonal_means(data, S))

        ds.close()

    if not times:
        print('No HRRR data found for the requested WY range.')
        return

    time_arr = np.concatenate(times)
    data_vars = {
        v: xr.DataArray(np.concatenate(accum[v], axis=0), dims=['time', 'divide_id'])
        for v in HRRR_VARS if accum[v]
    }
    out_ds = xr.Dataset(data_vars, coords={'time': time_arr, 'divide_id': hru_ids})

    if out_zarr.exists():
        shutil.rmtree(out_zarr)
    out_ds.to_zarr(out_zarr, zarr_format=2)
    print(f'Saved hrrr.zarr ({len(time_arr)} days x {len(hru_ids)} HRUs) → {out_zarr}')


if __name__ == '__main__':
    main()
