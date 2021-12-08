"""
Download images from NCSA for a given field and season
"""

import argparse
import glob
import os
import sys

os.chdir('..')

# get field, season, and progress info
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--field",
                    type=str,
                    help="Like X1, X2, E1, etc.",
                    required=True)
parser.add_argument("--season",
                    type=str,
                    help="Like SV, Y1, etc.",
                    required=True)
parser.add_argument("--check_progress",
                    action="store_true")
parser.add_argument("--be_nice",
                    action="store_true",
                    help="Only use 5 nodes in parallel")

args = parser.parse_args()

# Validate arguments
allowed_seasons = ["SV", "Y1", "Y2", "Y3", "Y4", "Y5"]  # GLOBAL
if args.season not in [""] + allowed_seasons:
    print("--season is invalid, must be " + ', '.join(allowed_seasons))
    sys.exit()
allowed_fields = ["E1", "E2", "C1", "C2", "C3", "S1", "S2", "X1", "X2", "X3"]  # GLOBAL
if args.field not in [""] + allowed_fields:
    print("--field is invalid, must be " + ', '.join(allowed_fields))
    sys.exit()

def _trim_list_file(listfile, season, field):
    # Trim to only images that aren't already downloaded (in case of restart)
    f = open(listfile, 'r')
    lines = f.readlines()
    f.close()
    
    files_to_do = []
    for line in lines:
        name = line.split('/')[-1].strip()
        if not os.path.exists(f"images/{season}/{name}"):
            files_to_do.append(line)

    if len(files_to_do) != 0:
        node = listfile.split('/')[-1].split('_')[0]
        list_file_name = f"images/{season}/{node}_{field}_wgetlist_temp.txt"
        f = open(list_file_name, 'w+')
        f.writelines(files_to_do)
        f.close()
        return list_file_name.split('/')[-1]
    
    return 'SKIP'

def _get_nodes(season, field):
    files = glob.glob(f"wget_lists/{season}/*_{field}_wgetlist.txt")
    nodes = {x.split('/')[-1].split('_')[0] : _trim_list_file(x, season, field) for x in files}
    return nodes

def download(args):
    """
    Read download lists and issue wget commands over ssh
    """
    nodes = _get_nodes(args.season, args.field)

    counter = 0
    for node, listfile in nodes.items():

        if listfile == 'SKIP':
            continue

        counter += 1
        if args.be_nice:
            if counter > 5:
                break
        
        command = (f'ssh rmorgan@{node}.fnal.gov ' + 
                   '"source /data/des81.b/data/stronglens/setup.sh && ' +   
                   f'cd /data/des81.b/data/stronglens/DEEP_FIELDS/PRODUCTION/images/{args.season}/ && ' +
                   f'wget --no-check-certificate -i {listfile} " &')

        os.system(command)


def check_progress(args):
    
    nodes = _get_nodes(args.season, args.field)

    # get total number of images
    num_images = {}
    for node in nodes:
        filename = f"{node}_{args.field}_wgetlist.txt"
        f = open(f"wget_lists/{args.season}/{filename}", 'r')
        lines = f.readlines()
        num_images[node] = {'TOTAL': len(lines), 'FILES': set([x.split('/')[-1].strip() for x in lines])}
        f.close()
    
    # get number of images downloaded
    all_files = set([x.split('/')[-1] for x in glob.glob(f"images/{args.season}/*.fz")])
    for node in nodes:
        num_images[node]['DONE'] = len(all_files.intersection(num_images[node]['FILES']))

    # print report
    running_total, running_done = 0, 0
    for node in nodes:
        print(node, '\tTotal:', num_images[node]['TOTAL'], '\tDone:', num_images[node]['DONE'], '\tRemaining:', num_images[node]['TOTAL'] - num_images[node]['DONE'])
        running_total += num_images[node]['TOTAL']
        running_done += num_images[node]['DONE']
    
    progress = running_done / running_total * 100
    print('\nProgress: %.1f %%\n' %progress)


# Main script action
if args.check_progress:
    check_progress(args)
else:
    # Make iamge directory if necessary
    if not os.path.exists('images/'):
        os.system("cp -r wget_lists images")
    else:
        files = glob.glob("images/*/*.fz")
        seasons = set([x.split('/')[1] for x in files])
        for season in seasons:
            if season != args.season:
                print(f"Existing images found in images/{season}/. Delete them before downloading new images to conserve disk space")
                sys.exit()

    download(args)
