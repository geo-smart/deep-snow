from pathlib import Path

from deep_snow.inputs import get_default_hill_pptwt_cache_path, get_default_hill_td_cache_path


PACKAGE_ROOT = Path(__file__).resolve().parent
RESOURCE_ROOT = PACKAGE_ROOT / "resources"
WEIGHTS_ROOT = RESOURCE_ROOT / "weights"
POLYGONS_ROOT = RESOURCE_ROOT / "polygons"


DEFAULT_MODEL_FILENAMES = [
    "ResDepth_lr0.000457131171011064_weightdecay0.00010523970398286011_epochs62_mintestloss0.00091",
    "ResDepth_lr0.00025036613931876504_weightdecay0.00020109428801183744_epochs63_mintestloss0.00091",
    "ResDepth_lr0.0001572907262097884_weightdecay0.00013101368652881237_epochs98_mintestloss0.00090",
    "ResDepth_lr0.00011563677025564128_weightdecay0.0003567649258551211_epochs63_mintestloss0.00092",
    "ResDepth_lr0.00026633575524604445_weightdecay8.770493085204089e-05_epochs85_mintestloss0.00090",
]


def get_packaged_model_path(filename):
    return str((WEIGHTS_ROOT / filename).resolve())


def get_default_model_paths():
    return [get_packaged_model_path(filename) for filename in DEFAULT_MODEL_FILENAMES]


def get_default_single_model_path():
    return get_default_model_paths()[2]


def get_default_land_path():
    return str((POLYGONS_ROOT / "ne_50m_land.shp").resolve())


def get_default_hill_pptwt_path():
    return str(Path(get_default_hill_pptwt_cache_path()).resolve())


def get_default_hill_td_path():
    return str(Path(get_default_hill_td_cache_path()).resolve())
