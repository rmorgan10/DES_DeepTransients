"""Run a deeplenstronomy simulation.

deeplenstronomy will inject simulated lensed SNe-Ia and SNe-CC into cutouts
from the PROCESSED/TRAINING_A directory. Output will be placed in the top-
level SIMULATIONS directory.

To run this script, you must be in the deeplens conda environment.
"""

import argparse
import os

from deeplenstronomy.deeplenstronomy import make_dataset

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--cutout_names", type=str,
    help="Comma-delimited list of cutouts to simulate.")
parser_args = parser.parse_args()

BASE_PATH = "/data/des81.b/data/stronglens/DEEP_FIELDS"

for cutout_name in parser_args.cutout_names.split(','):

    CONFIG_FILE = f"{BASE_PATH}/PROCESSED/TRAINING_A/{cutout_name}/dl_config.yaml"

    # Generate simulations.
    make_dataset(
        config=CONFIG_FILE,
        verbose=False,
        store_in_memory=False,
        save_to_disk=True,
        return_planes=True,
    )

    # Collect necessary files for reproduction, overwrite old sims.
    if os.path.exists(f"{BASE_PATH}/SIMULATIONS/{cutout_name}"):
        os.system(f"rm -rf {BASE_PATH}/SIMULATIONS/{cutout_name}")
    os.system(f"mv {cutout_name} {BASE_PATH}/SIMULATIONS")

    
