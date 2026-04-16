"""
Extract PRISM 800m daily forcings per HRU via pixel-weighted zonal means.

Data source: base_loc/PRISM/800m/{WY}.zarr
    dims: (time, lat, lon), EPSG:4326
    variables: ppt, tmean, tmax, tmin, tdmean, vpdmin, vpdmax

Output: model_domain_{name}/prism.zarr  (time, divide_id)

Usage:
    python 04_extract_prism.py --domain sierra_nevada
    python 04_extract_prism.py --domain sierra_nevada --wy_start 2014 --wy_end 2020
"""
import argparse
import shutil
import numpy as np
import xarray as xr
from scipy.sparse import csr_matrix
from domain_utils import domain_dir, load_hrus, domain_bbox, BASE_LOC

PRISM_DIR  = BASE_LOC / 'PRISM' / '800m'
PRISM_VARS = ['ppt', 'tmean', 'tmax', 'tmin', 'tdmean', 'vpdmin', 'vpdmax']


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
    hru  = flat[pix] - 1   # 0-based
    return csr_matrix((np.ones(len(pix)), (pix, hru)), shape=(len(flat), n_hrus))


def zonal_means(data: np.ndarray, S: csr_matrix) -> np.ndarray:
    """
    Pixel-weighted mean per HRU for all time steps via sparse matrix multiply.
    data : (n_time, n_lat, n_lon)
    S    : (n_pixels, n_hrus) sparse, built by build_weight_matrix
    Returns (n_time, n_hrus) float32, NaN where no valid pixels.
    """
    flat = data.reshape(len(data), -1).astype(np.float64)
    valid_data  = np.where(np.isfinite(flat), flat, 0.0)
    count_data  = np.isfinite(flat).astype(np.float64)
    sums   = S.T.dot(valid_data.T).T    # (n_time, n_hrus)
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

    out_zarr = domain_dir(args.domain) / 'prism.zarr'
    if out_zarr.exists() and not args.overwrite:
        print('prism.zarr already exists. Use --overwrite to redo.')
        return

    hrus = load_hrus(args.domain)
    bbox = domain_bbox(hrus)
    lon_min, lat_min, lon_max, lat_max = bbox

    S       = None
    hru_ids = None
    accum   = {v: [] for v in PRISM_VARS}
    times   = []

    for wy in range(args.wy_start, args.wy_end + 1):
        prism_path = PRISM_DIR / f'{wy}.zarr'
        if not prism_path.exists():
            print(f'  WY{wy}: {prism_path} not found, skipping')
            continue

        print(f'  WY{wy}...')
        ds = xr.open_zarr(prism_path).sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max),
        )

        if S is None:
            print('  Rasterizing HRUs on PRISM grid...')
            mask, hru_ids = rasterize_hrus(hrus, ds.lat.values, ds.lon.values)
            n_hrus = len(hru_ids)
            S = build_weight_matrix(mask, n_hrus)
            print(f'  {n_hrus} HRUs, {S.nnz} rasterized pixels')

        times.append(ds.time.values)
        for var in PRISM_VARS:
            data = ds[var].values.astype(np.float32)  # (n_time, n_lat, n_lon)
            accum[var].append(zonal_means(data, S))

        ds.close()

    if not times:
        print('No PRISM data found for the requested WY range.')
        return

    time_arr = np.concatenate(times)
    data_vars = {
        v: xr.DataArray(np.concatenate(accum[v], axis=0), dims=['time', 'divide_id'])
        for v in PRISM_VARS
    }
    out_ds = xr.Dataset(data_vars, coords={'time': time_arr, 'divide_id': hru_ids})

    if out_zarr.exists():
        shutil.rmtree(out_zarr)
    out_ds.to_zarr(out_zarr, zarr_format=2)
    print(f'Saved prism.zarr ({len(time_arr)} days x {len(hru_ids)} HRUs) → {out_zarr}')


if __name__ == '__main__':
    main()
