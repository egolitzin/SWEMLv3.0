"""
Subset the NWM NextGen hydrofabric to a named study domain.

Creates dataprocessing/model_domain_{name}/ and saves hrus.parquet with
HRU geometries and pre-computed attributes from the hydrofabric divide-attributes layer.

Usage:
    python 01_define_domain.py --domain sierra_nevada --bbox -121.5 36.0 -118.0 38.5
    python 01_define_domain.py --domain sierra_nevada --huc8 18030012 18040008 18040009
    python 01_define_domain.py --domain wus --vpuid 10L 10U 11 13 14 15 16 17 18
    python 01_define_domain.py --domain sierra_nevada --bbox -121.5 36.0 -118.0 38.5 --overwrite
"""
import argparse
import numpy as np
import geopandas as gpd
from domain_utils import domain_dir, HF_GPKG


def load_divides() -> gpd.GeoDataFrame:
    print('Loading hydrofabric divides + attributes...')
    div = gpd.read_file(HF_GPKG, layer='divides')
    da  = gpd.read_file(HF_GPKG, layer='divide-attributes')
    div = div.merge(da.drop(columns=['vpuid'], errors='ignore'), on='divide_id', how='left')
    return div.to_crs('EPSG:4326')


def subset_bbox(divides: gpd.GeoDataFrame, bbox: list[float]) -> gpd.GeoDataFrame:
    lon_min, lat_min, lon_max, lat_max = bbox
    return divides.cx[lon_min:lon_max, lat_min:lat_max].copy()


def subset_huc8(divides: gpd.GeoDataFrame, huc8_list: list[str]) -> gpd.GeoDataFrame:
    try:
        from pynhd import WBD
    except ImportError:
        raise ImportError('pynhd required for --huc8. Install with: pip install pynhd')
    import pandas as pd
    wbd   = WBD('huc8')
    polys = pd.concat([wbd.byid('huc8', h) for h in huc8_list])
    hrus  = gpd.sjoin(divides, polys[['huc8', 'geometry']], how='inner', predicate='intersects')
    hrus  = hrus.drop(columns='index_right').drop_duplicates(subset=['divide_id'])
    return hrus.copy()


def subset_vpuid(divides: gpd.GeoDataFrame, vpuid_list: list[str]) -> gpd.GeoDataFrame:
    return divides[divides['vpuid'].isin(vpuid_list)].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True, help='Name for this model domain')
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument('--bbox', nargs=4, type=float,
                     metavar=('LON_MIN', 'LAT_MIN', 'LON_MAX', 'LAT_MAX'))
    grp.add_argument('--huc8', nargs='+', metavar='HUC8')
    grp.add_argument('--vpuid', nargs='+', metavar='VPU')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    out_dir  = domain_dir(args.domain)
    out_file = out_dir / 'initial_hrus.parquet'

    if out_file.exists() and not args.overwrite:
        print(f"Domain '{args.domain}' already exists. Use --overwrite to redo.")
        return

    divides = load_divides()

    if args.bbox:
        hrus = subset_bbox(divides, args.bbox)
    elif args.huc8:
        hrus = subset_huc8(divides, args.huc8)
    elif args.vpuid:
        hrus = subset_vpuid(divides, args.vpuid)

    if len(hrus) == 0:
        raise ValueError('No HRUs found. Check bbox/HUC8/VPU values.')

    # pre-compute aspect sin/cos from the hydrofabric circular mean aspect
    if 'circ_mean.aspect' in hrus.columns:
        rad = np.deg2rad(hrus['circ_mean.aspect'])
        hrus['aspect_sin_hf'] = np.sin(rad)
        hrus['aspect_cos_hf'] = np.cos(rad)

    out_dir.mkdir(parents=True, exist_ok=True)
    hrus.to_parquet(out_file)
    print(f"Saved {len(hrus)} HRUs → {out_file}")
    print(f"  VPUs: {sorted(hrus['vpuid'].unique())}")
    print(f"  Bbox: {hrus.total_bounds.round(3).tolist()}")


if __name__ == '__main__':
    main()
