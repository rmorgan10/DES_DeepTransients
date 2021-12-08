# Iterate through a list of ccds and produce cutouts

import os
import sys

node = sys.argv[1]
season = sys.argv[2]
field = sys.argv[3]
maglim = sys.argv[4]
ccds = sys.argv[5:]

os.chdir("/data/des81.b/data/stronglens/DEEP_FIELDS/PRODUCTION/Y6_Bulk_Coadd_Cutouts/")
outdir = f"/data/des81.b/data/stronglens/DEEP_FIELDS/CUTOUTS/{season}/{field}/"

for ccd in ccds:
    rc = os.system(f'python cutout.py --season {season} --field {field} --ccd {ccd} --outdir {outdir} --maglim {maglim}')
    
    if rc == 0:
        os.system(f"touch /data/des81.b/data/stronglens/DEEP_FIELDS/PRODUCTION/job_sub/{field}_{season}_status/{field}_{season}_{ccd}.DONE")
    else:
        os.system(f"touch /data/des81.b/data/stronglens/DEEP_FIELDS/PRODUCTION/job_sub/{field}_{season}_status/{field}_{season}_{ccd}.ERROR")
