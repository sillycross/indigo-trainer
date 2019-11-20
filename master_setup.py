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
args = parser.parse_args()

with open('gcp_config.json') as json_file:
    data = json.load(json_file)
    assert('num_leaves' in data)
    NUM_LEAVES = data['num_leaves']
    assert('run_id' in data)
    RUN_ID = data['run_id']
    assert('zone' in data)
    GCP_ZONE = data['zone']
    
# setup the environment for trainer
#

# clone the model repo, initialize the correct branch

os.chdir(PROJECT_ROOT)

exit_code = os.system('python3 install_pkgs.py')
assert(exit_code == 0)

exit_code = os.system('git clone https://%s:%s@github.com/%s/%s.git' % (GITHUB_USER_NAME, args.cred, GITHUB_USER_NAME, MODEL_REPO_NAME))
assert(exit_code == 0)

os.chdir(PROJECT_ROOT + '/' + MODEL_REPO_NAME)

exit_code = os.system('git config user.name "Trainer Bot"')
assert(exit_code == 0)

# clone the Indigo LKM repo

os.chdir(PROJECT_ROOT)

exit_code = os.system('git clone https://github.com/sillycross/indigo-lkm')
assert(exit_code == 0)

# clone the tensorflow library and apply the patch

os.chdir(PROJECT_ROOT)

exit_code = os.system('git clone https://github.com/tensorflow/tensorflow')
assert(exit_code == 0)

os.chdir(PROJECT_ROOT + '/tensorflow')

exit_code = os.system('git checkout r1.14')
assert(exit_code == 0)

exit_code = os.system('git apply ../tensorflow-patch.txt')
assert(exit_code == 0)

exit_code = os.system('cp ../graph.config.pbtxt .')
assert(exit_code == 0)

# the default configure options seems to detect everything correctly
# sometimes yes returns broken pipe, but it's not a problem
exit_code = os.system('yes "" | ./configure')
assert(exit_code == 0)

# setup the model checkpoint repo, and generate the initial model

os.chdir(PROJECT_ROOT + '/' + MODEL_REPO_NAME)

exit_code = os.system('git reset --hard origin/master')
assert(exit_code == 0)

exit_code = os.system('git checkout -b %s' % RUN_ID)
assert(exit_code == 0)

exit_code = os.system('echo "-1" > version')
assert(exit_code == 0)

exit_code = os.system('git add version')
assert(exit_code == 0)

exit_code = os.system('git commit -m "Model branch = %s"' % RUN_ID)
assert(exit_code == 0)

exit_code = os.system('git push -u origin %s' % RUN_ID)
assert(exit_code == 0)

print('Building initial model...')

os.chdir(PROJECT_ROOT)

exit_code = os.system('echo "0" > %s/version' % MODEL_REPO_NAME)
assert(exit_code == 0)

exit_code = os.system('python3 build_initial_model.py')
assert(exit_code == 0)

exit_code = os.system('python3 build_pb_file.py')
assert(exit_code == 0)

exit_code = os.system('python3 build_lkm_file.py')
assert(exit_code == 0)

exit_code = os.system('python3 commit_model.py')
assert(exit_code == 0)

print('Everything seems to be OK!')

