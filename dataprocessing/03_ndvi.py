"""
Compute static canopy NDVI per HRU from MODIS MOD13A3.

We want a proxy for coniferous canopy density relevant to snow interception.
Using snow-season months (Nov-Apr): deciduous veg drops to near-zero NDVI during
this window, so the signal is dominated by evergreen/persistent vegetation only.
Multi-year mean over WY2014-2020 snow seasons gives a stable static attribute.

Data source: MODIS MOD13A3 v061, 1km monthly NDVI, via NASA Earthdata
Output: model_domain_{name}/ndvi.parquet

Attribute saved: canopy_ndvi (multi-year mean NDVI over Nov-Apr, 0-1 scaled)

Requires: pip install earthaccess exactextract

Files are cached in model_domain_{name}/_cache/modis/ and can be deleted after.

Usage:
    python ndvi.py --domain sierra_nevada
    python ndvi.py --domain sierra_nevada --months 11 12 1 2 3 4
"""
import argparse
import re
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import earthaccess
from pathlib import Path
from domain_utils import domain_dir, load_hrus, domain_bbox

MODIS_SHORT_NAME = 'MOD13A3'
MODIS_VERSION    = '061'
MODIS_NDVI_SCALE = 0.0001   # raw int16 -> [-0.2, 1.0]
MODIS_FILL_VALUE = -28672

# snow season: Nov-Apr, when deciduous canopy is absent
DEFAULT_MONTHS = [11, 12, 1, 2, 3, 4]


def download_modis(bbox: tuple, start_date: str, end_date: str, cache_dir: Path) -> list[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    earthaccess.login(strategy='netrc')
    results = earthaccess.search_data(
        short_name=MODIS_SHORT_NAME,
        version=MODIS_VERSION,
        temporal=(start_date, end_date),
        bounding_box=bbox,
    )
    if not results:
        raise RuntimeError('No MODIS MOD13A3 granules found for the specified domain/period.')
    print(f'  Found {len(results)} granules. Downloading...')
    files = earthaccess.download(results, local_path=str(cache_dir))
    return [Path(f) for f in files]


def parse_modis_date(fname: str) -> pd.Timestamp:
    """Parse acquisition date from MOD13A3 filename (e.g. MOD13A3.A2015032.h...)."""
    m = re.search(r'\.A(\d{7})\.', fname)
    if not m:
        raise ValueError(f'Cannot parse date from filename: {fname}')
    return pd.to_datetime(m.group(1), format='%Y%j')


def load_ndvi_stack(files: list[Path], bbox: tuple) -> xr.DataArray:
    """Load NDVI from HDF files, clip to bbox, stack into (time, y, x) DataArray."""
    arrays = []
    for f in sorted(files):
        try:
            date = parse_modis_date(f.name)
            da = rxr.open_rasterio(
                f'HDF4_EOS:EOS_GRID:"{f}":MOD_Grid_monthly_1km_VI:1 km monthly NDVI',
                masked=True,
            ).squeeze('band', drop=True)
            # MODIS MOD13A3 is in sinusoidal projection; reproject before bbox clip
            da = da.rio.reproject('EPSG:4326').rio.clip_box(*bbox)
            da = xr.where(da == MODIS_FILL_VALUE, np.nan, da * MODIS_NDVI_SCALE)
            da = da.expand_dims(time=[date])
            arrays.append(da)
        except Exception as e:
            print(f'  Warning: could not read {f.name}: {e}')
    return xr.concat(arrays, dim='time')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True)
    ap.add_argument('--months', nargs='+', type=int, default=DEFAULT_MONTHS,
                    help='Snow-season months to average over (default: Nov-Apr)')
    ap.add_argument('--start_wy', type=int, default=2014)
    ap.add_argument('--end_wy',   type=int, default=2020)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    out_file = domain_dir(args.domain) / 'ndvi.parquet'
    if out_file.exists() and not args.overwrite:
        print('ndvi.parquet already exists. Use --overwrite to redo.')
        return

    hrus  = load_hrus(args.domain)
    bbox  = domain_bbox(hrus)
    cache = domain_dir(args.domain) / '_cache' / 'modis'

    start = f'{args.start_wy - 1}-10-01'
    end   = f'{args.end_wy}-09-30'

    print(f'Downloading MOD13A3 ({start} to {end})...')
    files = download_modis(bbox, start, end, cache)

    print('Loading and stacking NDVI...')
    ndvi = load_ndvi_stack(files, bbox)

    # filter to snow season months and take multi-year mean
    ndvi_snow = ndvi.sel(time=ndvi.time.dt.month.isin(args.months))
    ndvi_mean = ndvi_snow.mean(dim='time', skipna=True)
    ndvi_mean = ndvi_mean.rio.write_crs('EPSG:4326')

    print('Running zonal statistics...')
    from exactextract import exact_extract
    df = exact_extract(ndvi_mean, hrus.to_crs('EPSG:4326'), ['mean'],
                        include_cols=['divide_id'], output='pandas')
    df = df.rename(columns={'mean': 'canopy_ndvi'})

    df.to_parquet(out_file, index=False)
    print(f'Saved NDVI attrs for {len(df)} HRUs → {out_file}')
    print(df['canopy_ndvi'].describe().round(3).to_string())


if __name__ == '__main__':
    main()
