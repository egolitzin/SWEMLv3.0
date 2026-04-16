"""
Join all pipeline outputs into NeuralHydrology-ready training inputs.

Reads from model_domain_{name}/:
    hrus.parquet          - HRU geometries and hydrofabric attrs
    terrain.parquet       - DEM zonal stats
    landcover.parquet     - NLCD fractions
    sturm.parquet         - Sturm snow class
    ndvi.parquet          - canopy NDVI
    {forcing}.zarr        - timeseries forcings (prism, hrrr, or aorc)
    targets.zarr          - UCLA SWE timeseries

Writes to model_domain_{name}/nh_inputs/{forcing_key}/ (NeuralHydrology GenericDataset layout):
    time_series/
        {divide_id}.nc    - per-HRU NetCDF with 'date' coordinate
    attributes/
        attributes.csv    - static attributes, one row per HRU

{forcing_key} is the sorted, underscore-joined list of forcings (e.g. hrrr, hrrr_prism, aorc_hrrr_prism).
The attributes/ folder is identical across forcing configs; it is written once per forcing_key run.

Usage:
    python 05_build_training_df.py --domain sierra_nevada --forcing prism
    python 05_build_training_df.py --domain sierra_nevada --forcing hrrr --overwrite
    python 05_build_training_df.py --domain sierra_nevada --forcing prism hrrr aorc
"""
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from domain_utils import domain_dir, load_hrus

# static parquets to join; each must have a 'divide_id' column
STATIC_FILES = [
    'terrain.parquet',
    'landcover.parquet',
    'sturm.parquet',
    'ndvi.parquet',
]

# columns from hrus.parquet to include as static attrs (hydrofabric pre-computed)
HF_STATIC_COLS = [
    'divide_id', 'areasqkm', 'vpuid',
    'mean.elevation', 'mean.slope', 'circ_mean.aspect',
    'aspect_sin_hf', 'aspect_cos_hf',
    'mean.impervious',
]


def load_static_attrs(domain: str) -> pd.DataFrame:
    d    = domain_dir(domain)
    hrus = load_hrus(domain)

    # select useful hydrofabric columns; drop geometry so it doesn't end up in attributes.csv
    keep = [c for c in HF_STATIC_COLS if c in hrus.columns]
    df   = pd.DataFrame(hrus[keep])

    for fname in STATIC_FILES:
        fpath = d / fname
        if not fpath.exists():
            print(f'  Warning: {fname} not found, skipping.')
            continue
        part = pd.read_parquet(fpath)
        # exactextract outputs quantile stats as quantile_25/quantile_75
        part = part.rename(columns={'quantile_25': 'elev_p25', 'quantile_75': 'elev_p75'})
        # one-hot encode sturm_class; fix columns to all 6 classes so encoding is
        # consistent across domains (a class absent in this domain gets all zeros)
        if 'sturm_class_name' in part.columns:
            all_classes = ['boreal_forest', 'ephemeral', 'maritime',
                           'montane_forest', 'prairie', 'tundra']
            dummies = pd.get_dummies(part['sturm_class_name'], prefix='sturm').astype(np.float32)
            for cls in [f'sturm_{c}' for c in all_classes]:
                if cls not in dummies.columns:
                    dummies[cls] = 0.0
            dummies = dummies[[f'sturm_{c}' for c in all_classes]]
            part = pd.concat([part.drop(columns=['sturm_class', 'sturm_class_name']), dummies], axis=1)
        df   = df.merge(part, on='divide_id', how='left')

    return df.set_index('divide_id')


def build_timeseries(domain: str, forcings: list[str]) -> xr.Dataset:
    """
    Load and merge forcing zarrs + targets zarr into a single Dataset
    with dimensions (time, divide_id).
    """
    d        = domain_dir(domain)
    datasets = []

    for forcing in forcings:
        fpath = d / f'{forcing}.zarr'
        if not fpath.exists():
            raise FileNotFoundError(
                f'{forcing}.zarr not found. Run the corresponding 04_extract_{forcing}.py first.'
            )
        datasets.append(xr.open_zarr(fpath))

    target_path = d / 'targets.zarr'
    if not target_path.exists():
        raise FileNotFoundError('targets.zarr not found. Run 04_extract_targets.py first.')
    datasets.append(xr.open_zarr(target_path))

    return xr.merge(datasets)


def write_nh_inputs(domain: str, forcings: list[str], overwrite: bool = False) -> None:
    d           = domain_dir(domain)
    forcing_key = '_'.join(sorted(forcings))
    data_dir    = d / 'nh_inputs' / forcing_key
    ts_dir      = data_dir / 'time_series'
    attrs_dir   = data_dir / 'attributes'
    ts_dir.mkdir(parents=True, exist_ok=True)
    attrs_dir.mkdir(parents=True, exist_ok=True)

    attrs_path = attrs_dir / 'attributes.csv'
    if not attrs_path.exists() or overwrite:
        print('Loading static attributes...')
        attrs = load_static_attrs(domain)
        zero_var = attrs.columns[(attrs.isna().all() | (attrs == 0).all())]
        if len(zero_var):
            print(f'  Dropping zero-variance columns: {list(zero_var)}')
            attrs = attrs.drop(columns=zero_var)
        attrs.to_csv(attrs_path)
        print(f'  Saved {len(attrs)} rows, {len(attrs.columns)} attrs → {attrs_path}')
    else:
        print('  attributes.csv exists, skipping (use --overwrite to redo)')

    print('Loading timeseries...')
    ts = build_timeseries(domain, forcings)
    # GenericDataset requires the time coordinate to be named 'date'
    ts = ts.rename({'time': 'date'})

    hru_ids = ts.coords['divide_id'].values
    print(f'Writing per-HRU NetCDF files ({len(hru_ids)} HRUs) → {forcing_key}/time_series/')

    for hru_id in hru_ids:
        out_nc = ts_dir / f'{hru_id}.nc'
        if out_nc.exists() and not overwrite:
            continue
        hru_ts = ts.sel(divide_id=hru_id).drop_vars('divide_id', errors='ignore')
        hru_ts.to_netcdf(out_nc)

    # write basin list (one divide_id per line) for NeuralHydrology config
    basin_file = data_dir / 'basins_all.txt'
    with open(basin_file, 'w') as f:
        f.write('\n'.join(str(b) for b in hru_ids))

    print(f'Done. Set data_dir: {data_dir}')
    print(f'  Forcings: {forcings}')
    print(f'  Time range: {str(ts.date.values[0])[:10]} to {str(ts.date.values[-1])[:10]}')
    print(f'  Variables: {list(ts.data_vars)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain',  required=True)
    ap.add_argument('--forcing', nargs='+', required=True,
                    choices=['prism', 'hrrr', 'aorc'],
                    help='One or more forcing datasets to include')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    write_nh_inputs(args.domain, args.forcing, args.overwrite)


if __name__ == '__main__':
    main()
