"""
Shared utilities for the dataprocessing pipeline.
All pipeline scripts import from here for consistent paths and domain loading.
"""
from pathlib import Path
import geopandas as gpd

REPO_ROOT = Path(__file__).parent.parent
BASE_LOC  = Path('/uufs/chpc.utah.edu/common/home/johnsonrc-group1')
HF_GPKG   = BASE_LOC / 'hydrofabric' / 'v2.2' / 'conus_nextgen.gpkg'


def domain_dir(name: str) -> Path:
    return REPO_ROOT / 'domains' / f'model_domain_{name}'


def load_hrus(name: str) -> gpd.GeoDataFrame:
    """Load the working (snow-filtered) HRU set. Requires 02_extract_targets.py to have run."""
    p = domain_dir(name) / 'hrus.parquet'
    if not p.exists():
        raise FileNotFoundError(
            f"hrus.parquet not found for domain '{name}'. "
            f"Run: python 02_extract_targets.py --domain {name}"
        )
    return gpd.read_parquet(p)


def load_initial_hrus(name: str) -> gpd.GeoDataFrame:
    """Load the full unfiltered HRU set written by 01_define_domain.py."""
    p = domain_dir(name) / 'initial_hrus.parquet'
    if not p.exists():
        raise FileNotFoundError(
            f"initial_hrus.parquet not found for domain '{name}'. "
            f"Run: python 01_define_domain.py --domain {name}"
        )
    return gpd.read_parquet(p)


def domain_bbox(hrus: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    """Return (lon_min, lat_min, lon_max, lat_max) in EPSG:4326."""
    b = hrus.to_crs('EPSG:4326').total_bounds
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])
