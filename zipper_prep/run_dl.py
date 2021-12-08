"""Run a deeplenstronomy simulation.

deeplenstronomy will inject simulated lensed SNe-Ia and SNe-CC into cutouts
from the PROCESSED/TRAINING_A directory. Output will be placed in the top-
level SIMULATIONS directory.

To run this script, you must be in the deeplens conda environment.
"""

import os

from deeplenstronomy.deeplenstronomy import make_dataset
from zippernet.data_utils import BASE_DATA_PATH

BASE_PATH = "/data/des81.b/data/stronglens/DEEP_FIELDS"
CONFIG_FILE = "dl_config.yaml"
AUXILLARY_FILES = [
    "source_light_profiles.txt",
    f"{BASE_PATH}/PROCESSED/TRAINING_A/map.txt",
    f"{BASE_PATH}/PROCESSED/TRAINING_A/g.fits",
    f"{BASE_PATH}/PROCESSED/TRAINING_A/r.fits",
    f"{BASE_PATH}/PROCESSED/TRAINING_A/i.fits",
    f"{BASE_PATH}/PROCESSED/TRAINING_A/z.fits",
]

# Generate simulations.
make_dataset(
    config=CONFIG_FILE,
    verbose=True,
    store_in_memory=False,
    save_to_disk=True,
    return_planes=True,
)

# Collect necessary files for reproduction, overwrite old sims.
if os.path.exists(f"{BASE_PATH}/SIMULATIONS/dl_sims"):
    os.system(f"rm -rf {BASE_PATH}/SIMULATIONS/dl_sims")
os.system(f"mv dl_sims {BASE_PATH}/SIMULATIONS")
for filename in [CONFIG_FILE] + AUXILLARY_FILES:
    os.system(f"cp {filename} {BASE_PATH}/SIMULATIONS")