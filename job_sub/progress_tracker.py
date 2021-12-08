# Track progress as cutout production runs

import glob
import os
import sys

field = sys.argv[1]
season = sys.argv[2]

# Parse jobs
jobs_file = f"{field}_{season}_status/{field}_{season}_jobs.log"
f = open(jobs_file, 'r')
lines = f.readlines()
f.close()

# Determine progress
node_dict = {}
for line in lines:
    node, outfile, maglim = line.split(',')
    
    if node not in node_dict:
        node_dict[node] = {'TOTAL': 0, 'DONE': 0, 'ERROR': 0}


    node_dict[node]['TOTAL'] += 1
        
    ccd = outfile.split('_')[-1].split('.')[0]

    prefix = f"/data/des81.b/data/stronglens/DEEP_FIELDS/PRODUCTION/job_sub/{field}_{season}_status/{field}_{season}_{ccd}"
    if os.path.exists(prefix + '.DONE'):
        node_dict[node]['DONE'] += 1
    if os.path.exists(prefix + '.ERROR'):
        node_dict[node]['ERROR'] += 1

# Print report
full_total = 0
full_done = 0
for node, info in node_dict.items():
    full_total += info['TOTAL']
    full_done += info['DONE'] +  info['ERROR']
    print(node, '\tTOTAL:', info['TOTAL'], '\tDONE:', info['DONE'], '\tERROR:', info['ERROR'], '\tRUNNING:', info['TOTAL'] - (info['DONE'] +  info['ERROR']))

progress = full_done / full_total * 100
print("\nProgress: %.1f %%\n" %progress)
