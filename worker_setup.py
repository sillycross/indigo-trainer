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
parser.add_argument("--branch", dest="branch", help="the branch prefix")
parser.add_argument("--worker_id", dest="worker_id", help="the worker id")
args = parser.parse_args()

assert(args.worker_id != None)

exit_code = os.system('git clone https://%s:%s@github.com/%s/%s.git' % (GITHUB_USER_NAME, args.cred, GITHUB_USER_NAME, MODEL_REPO_NAME))
assert(exit_code == 0)

os.chdir(PROJECT_ROOT + '/' + MODEL_REPO_NAME)

exit_code = os.system('git config user.name "Worker @ %s"' % args.worker_id)
assert(exit_code == 0)

exit_code = os.system('git checkout %s' % args.branch)
assert(exit_code == 0)

def push_remote_branch(branch_name):
	exit_code =  os.system('git push -u origin %s' % branch_name)
	assert(exit_code == 0)
	

