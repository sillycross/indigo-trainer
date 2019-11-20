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

gcp_zone_default = "us-central1-f"
parser = argparse.ArgumentParser()
parser.add_argument("--cred", dest="cred", help="the Github personal access token")
parser.add_argument("--num-leaves", dest="num_leaves", help="the number of leaves to use")
parser.add_argument("--zone", dest="zone", default=gcp_zone_default, help="the Google Cloud Compute zone to create instances")
parser.add_argument("--run-id", dest="run_id", help="If manually assigned, must be a globally unique ID for this train run. It will be used as instance group name, instance name prefix, and branch prefix in model repo. May only contain '-', 'a-z', '0-9'")

args = parser.parse_args()

assert(args.num_leaves != None)
args.num_leaves = int(args.num_leaves)
assert(args.num_leaves > 0)

os.chdir(PROJECT_ROOT)

# make sure that the git directory is clean 
#
#exit_code = os.system('[ -z "$(git status --untracked-files=no --porcelain)" ]')
#if (exit_code != 0):
#	print('Your git directory is not clean! Commit all changes before running this script!')
#	assert(False)
	
#exit_code = os.system("git status --untracked-files=no | grep 'Your branch is up to date with'")
#if (exit_code != 0):	# grep returns 1 on no matches and 2 on error
#	print('Your git branch is not up to date with remote! Push all changes before running this script!')
#	assert(False)

exit_code = os.system('git symbolic-ref --short HEAD > cur_branchname.txt')
assert(exit_code == 0)

with open('cur_branchname.txt') as f:
    trainer_repo_branch_name = f.read().splitlines()
    assert(len(trainer_repo_branch_name) == 1)
    trainer_repo_branch_name = trainer_repo_branch_name[0]
    
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
    while ('runid-%d' % k) in all_branches:
    	k += 1
    
    args.run_id = 'runid-%d' % k
    print('Assigned unique run ID: %s' % args.run_id)

instance_group_name = 'indigo-%s' % args.run_id
teardown_command = 'python3 gcp_env_teardown.py --run-id %s' % args.run_id
if (args.zone != gcp_zone_default):
    teardown_command += ' --zone %s' % args.zone
	
print('*********** Config ************')
print('Run ID: %s' % args.run_id)
print('Branch Name: %s' % trainer_repo_branch_name)
print('GCP Zone: %s' % args.zone)
print('Num Leaves: %d' % args.num_leaves)
print('*******************************')
print("!! Don't forget to run \n!!     %s\n!! to tear down environment after training!" % teardown_command)
print('*******************************')

# create instance group
#
print('Creating instance group..')
exit_code = os.system('gcloud compute instance-groups unmanaged create %s --zone %s' % (instance_group_name, args.zone))
assert(exit_code == 0)

def create_node(instance_group_name, instance_template, instance_name, instance_zone):
    cmd = """gcloud compute instances create %s \
--source-instance-template %s \
--project edgect-1155 \
--zone %s \
--scopes bigquery,cloud-platform,cloud-source-repos,\
cloud-source-repos-ro,compute-ro,compute-rw,\
datastore,default,gke-default,logging-write,\
monitoring,monitoring-write,pubsub,service-control,\
service-management,sql,sql-admin,storage-full,\
storage-ro,storage-rw,taskqueue,trace,userinfo-email""" % (instance_name, instance_template, instance_zone)
    exit_code = os.system(cmd)
    assert(exit_code == 0)
	
    cmd = """gcloud compute instance-groups unmanaged add-instances %s \
--instances %s --zone %s""" % (instance_group_name, instance_name, instance_zone)
    exit_code = os.system(cmd)
    assert(exit_code == 0)
    
# create each leaf and add to instance group 
#
for i in range(0, args.num_leaves):
    print('Creating leaf %d...' % i) 
    create_node(instance_group_name=instance_group_name,
                instance_template='indigo-lkm-leaf-4cpu', 
                instance_name='indigo-%s-leaf%d' % (args.run_id, i), 
                instance_zone=args.zone)

print('Creating master..')
create_node(instance_group_name=instance_group_name,
            instance_template='indigo-lkm-master', 
            instance_name='indigo-%s-master' % (args.run_id), 
            instance_zone=args.zone)

# Setup master and leaves 
# We need to clone the repo, checkout the corresponding branch and run the setup script


