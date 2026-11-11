"""
Join all pipeline outputs into NeuralHydrology-ready training or inference inputs.

Reads from model_domain_{name}/:
    hrus.parquet          - HRU geometries and hydrofabric attrs
    terrain.parquet       - DEM zonal stats
    landcover.parquet     - NLCD fractions
    sturm.parquet         - Sturm snow class
    ndvi.parquet          - canopy NDVI
    {forcing}.zarr        - timeseries forcings (prism, hrrr, or aorc)
    targets.zarr          - UCLA SWE timeseries (training mode only)

Writes to model_domain_{name}/nh_inputs/{forcing_key}/ (NeuralHydrology GenericDataset layout):
    time_series/
        {divide_id}.nc    - per-HRU NetCDF with 'date' coordinate
    attributes/
        attributes.csv    - static attributes, one row per HRU

With --inference, targets.zarr is excluded and outputs go to nh_inputs_inference/{forcing_key}/
so they never overwrite training inputs.

{forcing_key} is the sorted, underscore-joined list of forcings (e.g. hrrr, hrrr_prism, aorc_hrrr_prism).
The attributes/ folder is identical across forcing configs; it is written once per forcing_key run.

Usage:
    python 05_build_training_df.py --domain sierra_nevada --forcing prism
    python 05_build_training_df.py --domain sierra_nevada --forcing hrrr --overwrite
    python 05_build_training_df.py --domain sierra_nevada --forcing prism hrrr aorc
    python 05_build_training_df.py --domain aso --forcing prism --inference
    python 05_build_training_df.py --domain aso --forcing prism --inference --wy_start 2022 --wy_end 2024
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def parse_twi_dist(cell) -> float:
    """Extract weighted-mean TWI from a hydrofabric dist_N JSON cell."""
    if pd.isna(cell):
        return np.nan
    bins  = json.loads(cell) if isinstance(cell, str) else cell
    if not bins:
        return np.nan
    vals  = np.array([b['v'] for b in bins], dtype=float)
    freqs = np.array([b['frequency'] for b in bins], dtype=float)
    freqs /= freqs.sum()
    return float(np.dot(vals, freqs))


def load_static_attrs(domain: str) -> pd.DataFrame:
    d    = domain_dir(domain)
    hrus = load_hrus(domain)

    # select useful hydrofabric columns; drop geometry explicitly
    keep = [c for c in HF_STATIC_COLS if c in hrus.columns]
    df   = pd.DataFrame(hrus[keep]).drop(columns=['geometry'], errors='ignore')

    if 'dist_4.twi' in hrus.columns:
        df['twi_mean'] = hrus['dist_4.twi'].apply(parse_twi_dist).values
        n_nan = int(df['twi_mean'].isna().sum())
        if n_nan:
            df['twi_mean'] = df['twi_mean'].fillna(df['twi_mean'].mean())
            print(f'  Warning: twi_mean missing for {n_nan} HRUs; filled with domain mean.')
    else:
        print('  Warning: dist_4.twi not found in hrus.parquet, skipping twi_mean.')

    for fname in STATIC_FILES:
        fpath = d / fname
        if not fpath.exists():
            print(f'  Warning: {fname} not found, skipping.')
            continue
        part = pd.read_parquet(fpath)
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


def build_timeseries(domain: str, forcings: list[str], include_targets: bool = True) -> xr.Dataset:
    """
    Load and merge forcing zarrs (and optionally targets.zarr) into a single Dataset
    with dimensions (time, divide_id).
    Inner join on time so NetCDFs only span dates common to all inputs.
    Train/val/test splits are controlled entirely by the NeuralHydrology config dates.
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

    if include_targets:
        target_path = d / 'targets.zarr'
        if not target_path.exists():
            raise FileNotFoundError('targets.zarr not found. Run 04_extract_targets.py first.')
        datasets.append(xr.open_zarr(target_path))
    else:
        # NH invalidates samples with all-NaN targets even during inference.
        # Add a zero-filled dummy swe so samples are not skipped; swe_obs will
        # be zeros in results (NSE_ucla will be NaN, which is expected).
        ref = datasets[0]
        dummy_swe = xr.DataArray(
            np.zeros((len(ref.time), len(ref.divide_id)), dtype=np.float32),
            coords={'time': ref.time, 'divide_id': ref.divide_id},
            dims=['time', 'divide_id'],
            name='swe',
        )
        datasets.append(dummy_swe.to_dataset())

    ds = xr.merge(datasets, join='inner')

    # check for HRUs present in the reference dataset but dropped by the inner join
    ref_ids    = set(datasets[-1].divide_id.values)
    merged_ids = set(ds.divide_id.values)
    dropped    = ref_ids - merged_ids
    if dropped:
        missing_by_forcing = {}
        for i, forcing in enumerate(forcings):
            forcing_ids = set(datasets[i].divide_id.values)
            missing = ref_ids - forcing_ids
            if missing:
                missing_by_forcing[forcing] = len(missing)
        detail = ', '.join(f'{f}: {n} HRUs' for f, n in missing_by_forcing.items())
        ref_label = 'targets.zarr' if include_targets else f'{forcings[-1]}.zarr'
        raise ValueError(
            f'{len(dropped)} HRUs in {ref_label} are missing from other forcing zarr(s) '
            f'({detail}). Re-run the corresponding 04_extract_*.py scripts with --overwrite.'
        )

    t0, t1 = str(ds.time.values[0])[:10], str(ds.time.values[-1])[:10]
    print(f'  Merged time range (inner join): {t0} to {t1} ({len(ds.time)} days)')
    return ds


def write_nh_inputs(domain: str, forcings: list[str], overwrite: bool = False,
                    inference: bool = False,
                    wy_start: int | None = None, wy_end: int | None = None) -> None:
    d           = domain_dir(domain)
    forcing_key = '_'.join(sorted(forcings))
    inputs_root = 'nh_inputs_inference' if inference else 'nh_inputs'
    data_dir    = d / inputs_root / forcing_key
    ts_dir      = data_dir / 'time_series'
    attrs_dir   = data_dir / 'attributes'
    ts_dir.mkdir(parents=True, exist_ok=True)
    attrs_dir.mkdir(parents=True, exist_ok=True)

    attrs_path = attrs_dir / 'attributes.csv'
    if not attrs_path.exists() or overwrite:
        print('Loading static attributes...')
        attrs = load_static_attrs(domain)
        # fill columns with partial NaN (e.g. metrics undefined for flat basins) with column mean
        nan_cols = [c for c in attrs.columns if attrs[c].isna().any()]
        if nan_cols:
            n_affected = int(attrs[nan_cols].isna().any(axis=1).sum())
            print(f'  Warning: {n_affected} HRUs have NaN in {len(nan_cols)} column(s); filling with column median')
            for col in nan_cols:
                n_nan = int(attrs[col].isna().sum())
                attrs[col] = attrs[col].fillna(attrs[col].median())
                print(f'    {col}: {n_nan} NaN values filled')
        # after filling, any remaining NaN means extraction failed entirely for some HRUs
        nan_hrus = attrs.index[attrs.isna().any(axis=1)]
        if len(nan_hrus):
            raise ValueError(
                f'{len(nan_hrus)} HRUs in hrus.parquet have NaN static attributes. '
                f'Re-run 03_terrain.py, 03_landcover.py, 03_sturm_class.py, 03_ndvi.py '
                f'with --overwrite to cover the current HRU set.'
            )
        numeric  = attrs.select_dtypes(include='number')
        zero_var = numeric.columns[(numeric.isna().all() | (numeric == 0).all())]
        if len(zero_var):
            print(f'  Dropping zero-variance columns: {list(zero_var)}')
            attrs = attrs.drop(columns=zero_var)
        attrs.to_csv(attrs_path)
        print(f'  Saved {len(attrs)} rows, {len(attrs.columns)} attrs → {attrs_path}')
    else:
        print('  attributes.csv exists, skipping (use --overwrite to redo)')

    print('Loading timeseries...')
    ts = build_timeseries(domain, forcings, include_targets=not inference)

    if wy_start is not None or wy_end is not None:
        # include one full WY of warmup before wy_start to support up to a 365-day lookback
        t0 = f'{wy_start - 2}-10-01' if wy_start is not None else None
        t1 = f'{wy_end}-09-30'       if wy_end   is not None else None
        ts = ts.sel(time=slice(t0, t1))
        if len(ts.time) == 0:
            raise ValueError(
                f'Date slice {t0} to {t1} produced an empty time series. '
                f'Check that the forcing zarrs cover this period.'
            )
        actual_t0 = str(ts.time.values[0])[:10]
        if wy_start is not None and actual_t0 > t0:
            intended = (pd.Timestamp(f'{wy_start - 1}-10-01') - pd.Timestamp(t0)).days
            actual   = (pd.Timestamp(f'{wy_start - 1}-10-01') - pd.Timestamp(actual_t0)).days
            print(f'  Warning: only {actual} days of warmup before WY{wy_start} ({intended} requested); zarr starts at {actual_t0}')
        print(f'  Sliced to WY range (with warmup): {actual_t0} to {str(ts.time.values[-1])[:10]}')

    # GenericDataset requires the time coordinate to be named 'date'
    ts = ts.rename({'time': 'date'})

    hru_ids = ts.coords['divide_id'].values
    print(f'Writing per-HRU NetCDF files ({len(hru_ids)} HRUs) → {forcing_key}/time_series/')

    # materialize all data into RAM before the loop so each sel() is a numpy
    # slice rather than a zarr chunk read (one big read vs N_HRU small reads)
    print('  Loading all timeseries data into memory...')
    ts = ts.load()
    print('  Done loading.')

    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))

    def _write_hru(hru_id):
        out_nc = ts_dir / f'{hru_id}.nc'
        if out_nc.exists() and not overwrite:
            return
        hru_ts = ts.sel(divide_id=hru_id).drop_vars('divide_id', errors='ignore')
        hru_ts.to_netcdf(out_nc)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_write_hru, hid): hid for hid in hru_ids}
        done = 0
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 1000 == 0:
                print(f'  {done}/{len(hru_ids)} HRUs written')

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
                    choices=['prism','hrrr','aorc','gridmet','era5land','nldas'],
                    help='One or more forcing datasets to include')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--inference', action='store_true',
                    help='Omit targets.zarr; write to nh_inputs_inference/ instead of nh_inputs/')
    ap.add_argument('--wy_start', type=int, default=None,
                    help='First water year to include; one prior WY is prepended for LSTM warmup')
    ap.add_argument('--wy_end',   type=int, default=None,
                    help='Last water year to include (e.g. 2024 -> through 2024-09-30)')
    args = ap.parse_args()

    write_nh_inputs(args.domain, args.forcing, args.overwrite, args.inference,
                    args.wy_start, args.wy_end)


if __name__ == '__main__':
    main()
