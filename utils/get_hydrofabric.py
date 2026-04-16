"""
get_hydrofabric.py
Download the NWM NextGen hydrofabric (v2.2) from the lynker-spatial proxy
and provide utilities to subset it by bounding box or HUC8 list.

Authentication: bearer token from https://proxy.lynker-spatial.com/token
Store the token in ~/.lynker_spatial_token or set LYNKER_SPATIAL_TOKEN env var.

Data stored at:
    base_loc/hydrofabric/v2.2/conus_nextgen.gpkg     (4.4 GB, all CONUS)
    base_loc/hydrofabric/v2.2/conus_reference.gpkg   (4.6 GB, reference fabric)
    base_loc/hydrofabric/v2.2/hydrolocations.gpkg    (12 MB,  gauge/poi locations)

Layers inside conus_nextgen.gpkg:
    divides     - NWM catchment polygons (HRUs), each is one NeuralHydrology entity
    flowpaths   - stream flowpath geometries
    network     - topology table (toid, fromid, NHD COMID, VPU, etc.)
    nexus       - nexus/junction points

Usage
-----
# One-time setup: save your token (get it from https://proxy.lynker-spatial.com/token)
    echo "YOUR_TOKEN" > ~/.lynker_spatial_token
    chmod 600 ~/.lynker_spatial_token

# Download (run once; skips files that already exist):
    python utils/get_hydrofabric.py

# Subset from within Python:
    from utils.get_hydrofabric import load_hydrofabric, subset_by_bbox, subset_by_huc8

    divides = load_hydrofabric('divides')
    sn = subset_by_bbox(divides, bbox=(-121.5, 36.0, -118.0, 38.5))   # Sierra Nevada
    kin = subset_by_huc8(divides, huc8s=['18030012', '18040008', '18040009'])
"""

import os
import subprocess
from pathlib import Path

import geopandas as gpd
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_LOC   = Path('/uufs/chpc.utah.edu/common/home/johnsonrc-group1')
HF_DIR     = BASE_LOC / 'hydrofabric' / 'v2.2'

# Proxy base URL - authenticated HTTPS endpoint for downloads
PROXY_BASE = 'https://proxy.lynker-spatial.com/hydrofabric/v2.2/conus'

# Files to download: filename -> proxy path
DOWNLOAD_FILES = {
    'conus_nextgen.gpkg'   : f'{PROXY_BASE}/conus_nextgen.gpkg',
    'conus_reference.gpkg' : f'{PROXY_BASE}/conus_reference.gpkg',
    'hydrolocations.gpkg'  : f'{PROXY_BASE}/hydrolocations.gpkg',
}

# WUS VPUs (used for parquet reference layer downloads)
WUS_VPUS = ['10L', '10U', '11', '13', '14', '15', '16', '17', '18']

TOKEN_FILE = Path.home() / '.lynker_spatial_token'


# ── Auth ───────────────────────────────────────────────────────────────────────

def get_token() -> str:
    """Return the lynker-spatial bearer token.

    Reads from LYNKER_SPATIAL_TOKEN env var first, then ~/.lynker_spatial_token.
    Get your token by logging in at https://proxy.lynker-spatial.com and visiting
    https://proxy.lynker-spatial.com/token.
    """
    token = os.environ.get('LYNKER_SPATIAL_TOKEN', '').strip()
    if token:
        return token
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
        if token:
            return token
    raise RuntimeError(
        'No lynker-spatial token found. '
        'Visit https://proxy.lynker-spatial.com/token to get your token, then run:\n'
        f'  echo "YOUR_TOKEN" > {TOKEN_FILE}\n'
        '  chmod 600 ~/.lynker_spatial_token\n'
        'or set the LYNKER_SPATIAL_TOKEN environment variable.'
    )


# ── Download ──────────────────────────────────────────────────────────────────

def _wget(url: str, dest: Path, token: str) -> None:
    """Download url to dest using wget with bearer token auth. Resumes partial downloads."""
    subprocess.run(
        [
            'wget', '-c',
            '--header', f'Authorization: Bearer {token}',
            '-O', str(dest),
            url,
        ],
        check=True,
    )


def download_hydrofabric(
    files: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    """
    Download hydrofabric GeoPackage files from the lynker-spatial proxy.

    Parameters
    ----------
    files    : filenames to download (keys in DOWNLOAD_FILES).
               Defaults to ['conus_nextgen.gpkg', 'hydrolocations.gpkg'].
    overwrite: re-download even if the local file already exists.
    """
    if files is None:
        files = ['conus_nextgen.gpkg', 'hydrolocations.gpkg']

    token = get_token()
    HF_DIR.mkdir(parents=True, exist_ok=True)

    for fname in files:
        if fname not in DOWNLOAD_FILES:
            raise ValueError(f"Unknown file '{fname}'. Options: {list(DOWNLOAD_FILES)}")

        dest = HF_DIR / fname
        url  = DOWNLOAD_FILES[fname]

        if dest.exists() and not overwrite:
            print(f'[hydrofabric] {fname} already exists, skipping.')
            continue

        print(f'[hydrofabric] Downloading {fname} ...')
        _wget(url, dest, token)
        print(f'[hydrofabric] Saved -> {dest}  ({dest.stat().st_size / 1e9:.2f} GB)')


def download_reference_parquet(
    vpus: list[str] | None = None,
    layers: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    """
    Download WUS VPU-partitioned GeoParquet files for the reference fabric.

    Parameters
    ----------
    vpus   : VPU IDs to download.  Defaults to WUS_VPUS.
    layers : Reference layer names.  Defaults to ['reference_divides',
             'reference_flowpaths', 'reference_network'].
    overwrite : re-download if file already exists.
    """
    if vpus is None:
        vpus = WUS_VPUS
    if layers is None:
        layers = ['reference_divides', 'reference_flowpaths', 'reference_network']

    token = get_token()

    for layer in layers:
        for vpu in vpus:
            dest_dir = HF_DIR / 'reference' / layer / f'vpuid={vpu}'
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / 'part-0.parquet'
            url  = f'{PROXY_BASE}/reference/{layer}/vpuid={vpu}/part-0.parquet'

            if dest.exists() and not overwrite:
                print(f'[hydrofabric] {layer}/vpuid={vpu} exists, skipping.')
                continue

            print(f'[hydrofabric] Downloading {layer}/vpuid={vpu} ...')
            _wget(url, dest, token)


# ── Load / subset ─────────────────────────────────────────────────────────────

def load_hydrofabric(
    layer: str = 'divides',
    gpkg: str = 'conus_nextgen.gpkg',
) -> gpd.GeoDataFrame:
    """
    Load a layer from the CONUS hydrofabric GeoPackage.

    Parameters
    ----------
    layer : GeoPackage layer name.  One of:
            'divides', 'flowpaths', 'network', 'nexus'
    gpkg  : Which file to open.  One of:
            'conus_nextgen.gpkg' (default) or 'conus_reference.gpkg'

    Returns
    -------
    GeoDataFrame with the full CONUS layer.
    """
    path = HF_DIR / gpkg
    if not path.exists():
        raise FileNotFoundError(
            f'{path} not found. Run download_hydrofabric() first.'
        )
    return gpd.read_file(path, layer=layer)


def subset_by_bbox(
    gdf: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    """
    Spatial subset using a bounding box.

    Parameters
    ----------
    gdf  : GeoDataFrame (e.g. from load_hydrofabric).
    bbox : (lon_min, lat_min, lon_max, lat_max) in WGS-84 / EPSG:4326.

    Returns
    -------
    Filtered GeoDataFrame (rows whose geometry intersects the bbox).
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    from shapely.geometry import box as shapely_box
    aoi = shapely_box(lon_min, lat_min, lon_max, lat_max)

    gdf_wgs = gdf.to_crs('EPSG:4326') if gdf.crs and gdf.crs.to_epsg() != 4326 else gdf
    return gdf_wgs[gdf_wgs.intersects(aoi)].copy()


def subset_by_huc8(
    gdf: gpd.GeoDataFrame,
    huc8s: list[str],
    huc_col: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Filter catchments to those belonging to specified HUC8 basins.

    The NWM nextgen divides layer carries a 'huc8' column (or it can be
    derived from the 'id' field which is prefixed with the HUC8 code).
    Pass huc_col explicitly if the column name differs.

    Parameters
    ----------
    gdf     : GeoDataFrame of catchment divides.
    huc8s   : List of 8-digit HUC codes, e.g. ['18030012', '18040008', '18040009'].
    huc_col : Column name containing the HUC8 code.  Auto-detected if None.

    Returns
    -------
    Filtered GeoDataFrame.
    """
    if huc_col is None:
        # Try common column names
        for candidate in ('huc8', 'HUC8', 'reachcode'):
            if candidate in gdf.columns:
                huc_col = candidate
                break

    if huc_col and huc_col in gdf.columns:
        return gdf[gdf[huc_col].isin(huc8s)].copy()

    # Fall back: derive from 'id' column (NWM ids are prefixed "cat-XXXXXXXX...")
    if 'id' in gdf.columns:
        mask = gdf['id'].str.extract(r'(\d{8})')[0].isin(huc8s)
        result = gdf[mask].copy()
        if len(result) > 0:
            return result

    raise ValueError(
        f"Cannot find a HUC8 column.  Available columns: {list(gdf.columns)}"
    )


def load_reference_parquet(
    layer: str = 'reference_divides',
    vpus: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """
    Load VPU-partitioned GeoParquet files for the reference fabric.
    Reads only the requested VPUs and concatenates.

    Parameters
    ----------
    layer : One of 'reference_divides', 'reference_flowpaths', 'reference_network'.
    vpus  : VPU IDs to load.  Defaults to all downloaded WUS VPUs.

    Returns
    -------
    GeoDataFrame.
    """
    if vpus is None:
        vpus = WUS_VPUS

    parts = []
    for vpu in vpus:
        path = HF_DIR / 'reference' / layer / f'vpuid={vpu}' / 'part-0.parquet'
        if not path.exists():
            raise FileNotFoundError(
                f'{path} not found. Run download_reference_parquet(vpus=["{vpu}"]) first.'
            )
        df = gpd.read_parquet(path)
        df['vpuid'] = vpu
        parts.append(df)

    return pd.concat(parts, ignore_index=True) if parts else gpd.GeoDataFrame()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Download NWM NextGen hydrofabric v2.2 from lynker-spatial S3.'
    )
    parser.add_argument(
        '--files', nargs='+',
        default=['conus_nextgen.gpkg', 'hydrolocations.gpkg'],
        choices=list(DOWNLOAD_FILES.keys()),
        help='GeoPackage files to download (default: nextgen + hydrolocations).',
    )
    parser.add_argument(
        '--reference_parquet', action='store_true',
        help='Also download WUS VPU-partitioned reference parquet files.',
    )
    parser.add_argument(
        '--vpus', nargs='+', default=WUS_VPUS,
        help=f'VPU IDs for parquet download (default: {WUS_VPUS}).',
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Re-download even if files already exist.',
    )
    args = parser.parse_args()

    download_hydrofabric(files=args.files, overwrite=args.overwrite)

    if args.reference_parquet:
        download_reference_parquet(vpus=args.vpus, overwrite=args.overwrite)

    print('\nDone. Data saved to:', HF_DIR)
    print('\nTo subset the Sierra Nevada catchments:')
    print("  from utils.get_hydrofabric import load_hydrofabric, subset_by_huc8")
    print("  divides = load_hydrofabric('divides')")
    print("  sn = subset_by_huc8(divides, huc8s=['18030012', '18040008', '18040009'])")
