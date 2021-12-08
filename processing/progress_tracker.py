"""Track profress of processing jobs."""

import os

from make_all_dl_inputs import OUTPUT_PATH

OUTPUT_PATH = "/data/des81.b/data/stronglens/DEEP_FIELDS/PROCESSED/TESTING"

job_file = 'status/make_dl_jobs.txt'
with open(job_file) as f:
    jobs = f.readlines()

progress_dict = {}
for job in jobs:
    node, cutout_name, args = [x.strip() for x in job.split(',')]

    if node not in progress_dict:
        progress_dict[node] = {'DONE': 0, 'TOTAL': 0}

    progress_dict[node]['TOTAL'] += 1
    if os.path.exists(f"{OUTPUT_PATH}/{cutout_name[0:-4]}/images.npy"):
        progress_dict[node]['DONE'] += 1


done_total, todo_total = 0, 0
for node in sorted(list(progress_dict.keys())):
    done = progress_dict[node]['DONE']
    done_total += done
    todo = progress_dict[node]['TOTAL'] - done
    todo_total += todo
    msg = f"{node}\tDONE: {done}\t\tTODO: {todo}\t\tTOTAL: {done + todo}"
    print(msg)

progress = round(done_total * 100 / (done_total + todo_total), 2)
print(f'\nProgress: {progress:.2f} %')
