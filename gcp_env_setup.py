import os
import argparse
import json
import threading
import os.path
from os import path
import time
import subprocess
import socket

from env_vars import *

parser = argparse.ArgumentParser()
parser.add_argument("--cred", dest="cred", help="the Github personal access token")
parser.add_argument("--num-leaves", dest="num_leaves", help="the number of leaves to use")
parser.add_argument("--zone", dest="zone", default="us-central1-f", help="the Google Cloud Compute zone to create instances")
parser.add_argument("--run-id", dest="run_id", help="If manually assigned, must be a globally unique ID for this train run. It will be used as instance group name, instance name prefix, and branch prefix in model repo.")

args = parser.parse_args()

assert(args.num_leaves != None)
args.num_leaves = int(args.num_leaves)
assert(args.num_leaves > 0)

os.chdir(PROJECT_ROOT)

if (args.run_id == None):
    print('Run ID not specified, automatically assigning unique ID to this run...')
    exit_code = os.system('git ls-remote --heads git@github.com:%s/%s.git > all_branchnames.txt' % (GITHUB_USER_NAME, MODEL_REPO_NAME))
    assert(exit_code == 0)
    with open('all_branchnames.txt') as f:
        content = f.read().splitlines()
        
    all_branches = []
    for line in content:
        values = line.split('\t')
        assert(len(values) == 2)
        x = 'refs/heads/'
        assert(values[1][0:len(x)] == x)
        all_branches.append(values[1][len(x):])
    print('Found following branches: ')
    print(all_branches)
    
    k = 0
    while ('runid_%d' % k) in all_branches:
    	k += 1
    
    args.run_id = 'runid_%d' % k
    print('Assigned unique run ID: %s' % args.run_id)

print('*******************************')
print('Run ID: %s' % args.run_id)
print('GCP Zone: %s' % args.zone)
print('Num Leaves: %d' % args.num_leaves)
print('*******************************')

# create instance group
#
instance_group_name = 'indigo_%s' % args.run_id

