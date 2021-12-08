import os
import sys

node = sys.argv[1]
season = sys.argv[2]
field = sys.argv[3]
maglim = sys.argv[4]
ccds = sys.argv[5:]
for ccd in ccds:
    os.system(f'touch /data/des81.b/data/stronglens/DEEP_FIELDS/PRODUCTION/job_sub/{field}_{season}_status/{node}_{season}_{field}_{ccd}.READY')
