"""Make a deeplenstronomy config file for each TRAINING_A cutout.

The trainging data for source injection is separated by cutout. This script
builds up a single set of training data from mutliple cutouts.
"""

import glob
import os
import random
from typing import List

import pandas as pd

BASE_PATH = "/data/des81.b/data/stronglens/DEEP_FIELDS/PROCESSED/TRAINING_A"


def make_config(cutout_name: str, seed: int, lines: List[str]) -> int:
  """Make a dl config file for a cutout.

  Update the base config file with a new random seed, number of images, and
  image paths so that each cutout can be an independent simulation.

  Args:
    cutout_name (str): Name of cutout (like X3_Y2_4).
    seed (int): Random seed for simulation (natural number).
    lines (List[str]): Contents of base config file.

  Returns:
    num_images for status tracking.
  """
  # Get number of images to simulate save image path.
  image_path = f"{BASE_PATH}/{cutout_name}"
  map_df = pd.read_csv(f"{image_path}/map.txt")
  num_images = len(map_df)

  if num_images == 0:
    os.system(f"touch {BASE_PATH}/{cutout_name}/EMPTY.SKIP")

  # Set seed, image path, and number of images in config file contents.
  out_lines = []
  for line in lines:
    if line.startswith("        SEED:"):
      out_lines.append(f"        SEED: {seed}\n")
    elif line.startswith("        SIZE:"):
      out_lines.append(f"        SIZE: {num_images}\n")
    elif line.startswith("        OUTDIR:"):
      out_lines.append(f"        OUTDIR: {cutout_name}\n")
    elif line.startswith("    PATH:"):
      out_lines.append(f"    PATH: {image_path}\n")
    else:
      out_lines.append(line)

  with open(f"{image_path}/dl_config.yaml", "w+") as f:
    f.writelines(out_lines)

  return num_images

    
if __name__ == "__main__":

  # Load base config file.
  f = open("dl_config.yaml")
  lines = f.readlines()
  f.close()

  # Make individual config files.
  cutout_dirs = [x for x in glob.glob(f'{BASE_PATH}/*') if os.path.isdir(x)]

  # Determine random seeds.
  seeds = list(range(len(cutout_dirs)))
  random.shuffle(seeds)

  num_images, empty_cutouts = 0, []
  for cutout_dir, seed in zip(cutout_dirs, seeds):
    cutout_name = cutout_dir.split('/')[-1]
    new_images = make_config(cutout_name, seed, lines)
    if new_images == 0:
      empty_cutouts.append(cutout_name)
    num_images += new_images

  # Print out stats
  print("Total image sets: ", num_images)
  print(f"Empty cutouts: ({len(empty_cutouts)})\n", empty_cutouts)