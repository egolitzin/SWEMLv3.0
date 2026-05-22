"""
Define a model domain from the union of NWM catchments covered by ASO flights.

Scans all raw ASO GeoTIFFs (50m SWE), reads each flight's spatial footprint via
rasterio metadata (no pixel reads), and finds all hydrofabric HRUs that intersect
at least one flight. The hydrofabric is loaded with an OGR bbox filter so only
the relevant portion of the CONUS file is read into memory.

Creates:
  domains/model_domain_{name}/initial_hrus.parquet  HRU geometries + attributes
  domains/model_domain_{name}/flight_index.parquet  one row per ASO tif

Usage:
  python 01_define_domain_aso.py
  python 01_define_domain_aso.py --domain aso --overwrite
"""
import argparse
import re
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

from domain_utils import domain_dir, HF_GPKG

ASO_DIR = Path(__file__).parent.parent / 'data' / 'ASO'


def scan_aso_tifs() -> list[Path]:
    tifs = []
    for yr_dir in sorted(ASO_DIR.iterdir()):
        if not (yr_dir.is_dir() and yr_dir.name.isdigit()):
            continue
        raw_dir = yr_dir / 'Raw_ASO_Data'
        if raw_dir.is_dir():
            tifs.extend(sorted(raw_dir.glob('*.tif')))
    return tifs


def parse_flight_date(stem: str) -> datetime | None:
    # Convention 1: stem ends with _YYYYMMDD (e.g. ASO_50M_SWE_USCATB_20130403)
    m = re.search(r'_(\d{8})$', stem)
    if m:
        try:
            return datetime.strptime(m.group(1), '%Y%m%d')
        except ValueError:
            pass
    # Convention 2: _YYYYMonDD in stem (e.g. ASO_Kings_2020Apr11-14_swe_50m)
    m = re.search(r'_(\d{4})(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\d+)',
                  stem, re.IGNORECASE)
    if m:
        year = m.group(1)
        mon  = m.group(2).capitalize()
        day  = m.group(3).split('-')[0]
        try:
            return datetime.strptime(f'{year}{mon}{int(day):02d}', '%Y%b%d')
        except ValueError:
            pass
    return None


def parse_basin_name(stem: str) -> str | None:
    # Convention 1: basin token appears just before trailing _YYYYMMDD
    m = re.search(r'_([^_]+)_\d{8}$', stem)
    if m:
        return m.group(1)
    # Convention 2: basin token appears just after ASO_ (letters only, stops before digit year)
    m = re.match(r'^ASO_([A-Za-z]+)_', stem)
    if m:
        return m.group(1)
    return None


def build_flight_index(tif_paths: list[Path]) -> gpd.GeoDataFrame:
    """Read bounds from each ASO tif (metadata only), return GeoDataFrame in EPSG:4326."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    target_crs = CRS.from_epsg(4326)
    records    = []
    n_failed   = 0

    for tif in tif_paths:
        try:
            with rasterio.open(tif) as src:
                bounds  = src.bounds
                src_crs = src.crs
            if src_crs is None:
                raise ValueError('no CRS defined')
            lon_min, lat_min, lon_max, lat_max = transform_bounds(
                src_crs, target_crs,
                bounds.left, bounds.bottom, bounds.right, bounds.top,
            )
        except Exception as e:
            print(f'  WARNING: skipping {tif.name}: {e}')
            n_failed += 1
            continue
        records.append({
            'tif_path':    str(tif),
            'basin':       parse_basin_name(tif.stem),
            'year':        int(tif.parent.parent.name),
            'flight_date': parse_flight_date(tif.stem),
            'lon_min':     lon_min,
            'lat_min':     lat_min,
            'lon_max':     lon_max,
            'lat_max':     lat_max,
            'geometry':    box(lon_min, lat_min, lon_max, lat_max),
        })

    if n_failed:
        print(f'  {n_failed} tifs could not be read and were skipped')
    return gpd.GeoDataFrame(records, crs='EPSG:4326')


def load_divides_bbox(bbox_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Load hydrofabric divides + attributes for HRUs within bbox_gdf."""
    import pyogrio
    # pyogrio requires bbox in the file's native CRS; read it to pass to gpd.read_file later
    file_crs  = pyogrio.read_info(HF_GPKG, layer='divides')['crs']
    bbox_native = tuple(bbox_gdf.to_crs(file_crs).total_bounds)
    lon_min, lat_min, lon_max, lat_max = bbox_gdf.total_bounds
    print(f'  divides (bbox: {lon_min:.2f},{lat_min:.2f} -> {lon_max:.2f},{lat_max:.2f})...')
    div = gpd.read_file(HF_GPKG, layer='divides', bbox=bbox_native).to_crs('EPSG:4326')
    print(f'  {len(div)} divides in bbox')

    print('  divide-attributes...')
    da = gpd.read_file(HF_GPKG, layer='divide-attributes')
    da = da[da['divide_id'].isin(set(div['divide_id']))].drop(columns=['vpuid'], errors='ignore')
    return div.merge(da, on='divide_id', how='left')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain',    default='aso',
                    help='Name for this model domain (default: aso)')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    out_dir   = domain_dir(args.domain)
    out_hrus  = out_dir / 'initial_hrus.parquet'
    out_index = out_dir / 'flight_index.parquet'

    if out_hrus.exists() and not args.overwrite:
        print(f"Domain '{args.domain}' already exists. Use --overwrite to redo.")
        return

    # 1. Read flight footprints from tif metadata
    print('Scanning ASO GeoTIFFs...')
    tifs    = scan_aso_tifs()
    print(f'  {len(tifs)} tifs found under {ASO_DIR}')
    flights = build_flight_index(tifs)
    print(f'  {len(flights)} flight footprints indexed')

    if flights.empty:
        raise RuntimeError(f'No readable ASO tifs found under {ASO_DIR}')

    # 2. Union all flight footprints into a single geometry for bbox + spatial join
    print('Computing ASO coverage union...')
    union_poly = unary_union(flights.geometry)
    union_gdf  = gpd.GeoDataFrame(geometry=[union_poly], crs='EPSG:4326')
    n_parts    = len(union_poly.geoms) if union_poly.geom_type == 'MultiPolygon' else 1
    lon_min, lat_min, lon_max, lat_max = union_poly.bounds
    print(f'  {n_parts} disjoint region(s), '
          f'extent {lon_min:.2f},{lat_min:.2f} -> {lon_max:.2f},{lat_max:.2f}')

    # 3. Load hydrofabric divides within the union bbox.
    print('Loading hydrofabric divides...')
    divides = load_divides_bbox(union_gdf)

    # 4. Precise spatial join against the union polygon
    print('Joining HRUs to ASO flight union...')
    hrus = gpd.sjoin(divides, union_gdf, how='inner', predicate='intersects')
    hrus = hrus.drop(columns='index_right').drop_duplicates(subset=['divide_id'])
    print(f'  {len(hrus)} HRUs intersect ASO coverage (of {len(divides)} in bbox)')

    if len(hrus) == 0:
        raise ValueError('No HRUs found. Check ASO_DIR and hydrofabric path.')

    # 5. Pre-compute aspect sin/cos from hydrofabric circular mean
    if 'circ_mean.aspect' in hrus.columns:
        rad = np.deg2rad(hrus['circ_mean.aspect'])
        hrus['aspect_sin_hf'] = np.sin(rad)
        hrus['aspect_cos_hf'] = np.cos(rad)

    out_dir.mkdir(parents=True, exist_ok=True)
    hrus.to_parquet(out_hrus)
    print(f'Saved {len(hrus)} HRUs -> {out_hrus}')
    print(f'  VPUs: {sorted(hrus["vpuid"].dropna().unique())}')
    print(f'  Bbox: {hrus.total_bounds.round(3).tolist()}')

    flights.to_parquet(out_index)
    n_dated  = flights['flight_date'].notna().sum()
    n_basin  = flights['basin'].notna().sum()
    print(f'Saved flight index ({len(flights)} flights, '
          f'{n_dated} with parsed dates, {n_basin} with parsed basins) -> {out_index}')


if __name__ == '__main__':
    main()
