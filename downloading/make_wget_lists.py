"""
Convert filepaths into a list file for wget
"""

import glob
import os

import pandas as pd

os.chdir('..')

# Make output directory
if os.path.exists('wget_lists/'):
    os.system('rm -rf wget_lists/')
os.mkdir('wget_lists/')
os.mkdir('wget_lists/SV/')
os.mkdir('wget_lists/Y1/')
os.mkdir('wget_lists/Y2/')
os.mkdir('wget_lists/Y3/')
os.mkdir('wget_lists/Y4/')
os.mkdir('wget_lists/Y5/')

# Establish nodes
des_nodes = ["des30", "des31", "des40", "des41", "des50",
             "des60", "des70", "des71", "des80", "des81", "des90", "des91"]

# Get metadata
md_files = glob.glob('metadata/*.csv')

# function to produce wget lists
def make_wget_list(nodes : list, df : pd.DataFrame) -> dict :
    """
    Make a wget list for each node from all paths in a dataframe, group by season
    """

    season_groups = df.groupby('SEASON')
    outdict = {node : {} for node in nodes}
    node_idx = 0
    for (season, season_df) in season_groups:

        for index, row in season_df.iterrows():

            if season not in outdict[nodes[node_idx]]:
                outdict[nodes[node_idx]][season] = []

            outdict[nodes[node_idx]][season].append(f"https://desar2.cosmology.illinois.edu/DESFiles/desarchive/{row['PATH']}/{row['FILENAME']}{row['COMPRESSION']}\n")

            node_idx += 1
            if node_idx == len(nodes):
                node_idx = 0

    return outdict

# function to write a text file
def write_text_file(lines : list, filename : str):
    """
    Write list of strings to text file named <filename>
    """
    f = open(filename, 'w+')
    f.writelines(lines)
    f.close()

# Loop through metadata and produce list files

for md_file in md_files:
    df = pd.read_csv(md_file)
    field = md_file.split('/')[1].split('_')[0]
    print(field)

    season_dict = make_wget_list(des_nodes, df)

    for node, info in season_dict.items():
        
        for season, lines in info.items():

            outfile = f"wget_lists/{season}/{node}_{field}_wgetlist.txt"
            write_text_file(lines, outfile)

