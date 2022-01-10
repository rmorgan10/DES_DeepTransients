"""Create deeplenstronomy inputs for all existing cutout files.

This script performs several tasks to create inputs for deeplenstronomy source
injection:
    (1) Split input data into training and testing datasets
    (2) Divide the training into A (signal) and B (backgorund) samples
    (3) Distribute make_dl_inputs.py jobs over DES nodes.

"""

import argparse
import glob
import os
import random
from typing import Any, List, Tuple

CUTOUT_PATH = "/data/des81.b/data/stronglens/DEEP_FIELDS/CUTOUTS"
OUTPUT_PATH = "/data/des81.b/data/stronglens/DEEP_FIELDS/PROCESSED"
data_dict = {
    'training': {'A': [], 'B': []},
    'testing': []
}
DES_NODES_a = ["des30", "des40"]
DES_NODES_b = ["des50", "des60"]
DES_NODES_test = [
    "des31", "des41", "des70", "des71", "des80", "des81", "des90", "des91"
]


def get_remaining_cutouts() -> List[str]:
    """Determine which available cutout files still need to be processed.

    Returns:
        A list of file names to process, split by A and B sufficies.
    """
    all_cutout_files = glob.glob(f"{CUTOUT_PATH}/*/*/*.npy")
    done_cutout_files = set(glob.glob(f"{OUTPUT_PATH}/TESTING/*/images.npy"))
    remaining_cutout_files = []

    for cutout_file in all_cutout_files:
        name = cutout_file.split('/')[-1][:-4]
        done_filename = f"{OUTPUT_PATH}/TESTING/{name}/images.npy"
        if done_filename not in done_cutout_files:
            remaining_cutout_files.append(cutout_file)

    return remaining_cutout_files

def split_training_testing(file_list: List[str]):
    """Split cutout files into training and testing.
    
    By default, 1% of images are used for training, and the A and B subsets
    of the training data are equivalent in size. The splitting is done randomly.
    The file names are sorted into the global variable `data_dict`.

    The X3, C3, and E2 fields have information needed for training, so they
    are preferentially used to select the training data.

    Args:
      file_list (List[str]): A list of file names to process.
    """

    def _split_list(
        items: List[Any], fraction: float) -> Tuple[List[Any], List[Any]]:
        """Randomly split a list into two parts.
        
        Args:
          items (List[Any]): A list to split into two parts.
          fraction (float): The fraction of total elements for the first part.

        Returns:
          Two lists composed of the elements of the input list.

        Raises:
          ValueError if fraction not in [0,1].
          ValueError if items is an empty list.
        """
        if not 0.0 < fraction < 1.0:
            raise ValueError("fraction must be between 0 and 1")
        if len(items) == 0:
            return [], []

        split_idx = int(fraction * len(items))
        items_ = items[:]
        random.shuffle(items_)
        return items_[:split_idx], items_[split_idx:]


    file_list_a, file_list_b = [], []
    for filename in file_list:
        if (filename.rfind('X3') != -1 or 
            filename.rfind('E2') != -1 or 
            filename.rfind('C3') != -1):
            file_list_a.append(filename)
        else:
            file_list_b.append(filename)

    training_a, leftovers = _split_list(file_list_a, 0.2)
    remaining_data = leftovers + file_list_b
    b_frac = max(0.001, len(training_a) / len(remaining_data))
    training_b, testing = _split_list(remaining_data, b_frac)

    data_dict['testing'] = testing
    data_dict['training']['A'] = training_a
    data_dict['training']['B'] = training_b

def queue_up_jobs() -> dict:
    """Produce a text file containing the jobs to be run.
    
    Returns:
      A list of jobs to distribute.
    """
    def _update_node_idx(idx: int, list_var: List[Any]) -> int:
        """Increment idx unless it has reached the end of list_var."""
        idx += 1
        if idx == len(list_var):
            idx = 0
        return idx

    def _get_cutout_name(filename: str) -> str:
        """Select just the local filename from a full file path.
        
        Args:
          filename (str): Path to a cutout file.
          
        Returns:
          The name of the cutout at a given file path.
        """
        return filename.split('/')[-1]
        
    # Distribute jobs on DES nodes.
    jobs = {}
    node_idx = 0
    for filename in data_dict['training']['A']:
        node = DES_NODES_a[node_idx]
        if node in jobs:
            jobs[node].append(f"{_get_cutout_name(filename)},--for_training_a\n")
        else:
            jobs[node] = [f"{_get_cutout_name(filename)},--for_training_a\n"]
        node_idx = _update_node_idx(node_idx, DES_NODES_a)

    node_idx = 0
    for filename in data_dict['training']['B']:
        node = DES_NODES_b[node_idx]
        if node in jobs:
            jobs[node].append(f"{_get_cutout_name(filename)},--for_training_b\n")
        else:
            jobs[node] = [f"{_get_cutout_name(filename)},--for_training_b\n"]
        node_idx = _update_node_idx(node_idx, DES_NODES_b)

    node_idx = 0
    for filename in data_dict['testing']:
        node = DES_NODES_test[node_idx]
        if node in jobs:
            jobs[node].append(f"{_get_cutout_name(filename)},\n")
        else:
            jobs[node] = [f"{_get_cutout_name(filename)},\n"]
        node_idx = _update_node_idx(node_idx, DES_NODES_test)

    # Save a text file for progress checking.
    with open("status/make_dl_jobs.txt", "w+") as f:
        for node, job_list in jobs.items():
            for job in job_list:
                f.write(f"{node},{job}")

    return jobs


if __name__ == "__main__":

    # Handle command-line arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggressive", action="store_true", 
        help="Run cutout jobs concurrently on each node.")
    parser_args = parser.parse_args()

    # Determine jobs to distribute.
    remaining_cutout_files = get_remaining_cutouts()
    split_training_testing(remaining_cutout_files)
    jobs = queue_up_jobs()

    # Submit jobs.
    for node, job_list in jobs.items():
        args = job_list[0].split(',')[-1].strip()
        cutout_names = ','.join([x.split(',')[0] for x in job_list])

        if parser_args.aggressive:
            for cutout_name in cutout_names.split(','):
                cmd = (
                    f'ssh rmorgan@{node}.fnal.gov ' +
                    '"source /data/des81.b/data/stronglens/setup.sh && '
                    'cd /data/des81.b/data/stronglens/DEEP_FIELDS/PRODUCTION/processing/ &&'
                    f'python make_dl_inputs.py --filenames {cutout_name} {args}" &'
                )
                os.system(cmd)

        else:
            cmd = (
                f'ssh rmorgan@{node}.fnal.gov ' +
                '"source /data/des81.b/data/stronglens/setup.sh && '
                'cd /data/des81.b/data/stronglens/DEEP_FIELDS/PRODUCTION/processing/ &&'
                f'python make_dl_inputs.py --filenames {cutout_names} {args}" &'
            )

            os.system(cmd)
        



    
