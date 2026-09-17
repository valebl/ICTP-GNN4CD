import os

def setup_cartopy(cartopy_data_dir=None):
    """
    Ensures CARTOPY_DATA_DIR is set before importing cartopy.
    """
    # 1. Determine the directory
    if cartopy_data_dir is None:
        # Default
        cartopy_data_dir = "/leonardo_work/ICT26_ESP/vblasone/cartopy/"

    # 2. Set env var ONLY if not already set
    if "CARTOPY_DATA_DIR" not in os.environ:
        os.environ["CARTOPY_DATA_DIR"] = cartopy_data_dir

    return
