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
parser.add_argument("--branch", dest="branch", help="the target branch to store the models")
args = parser.parse_args() 

assert(args.branch != None)

print('Initializing a new training branch at "%s"' % args.branch)

os.chdir(PROJECT_ROOT + '/' + MODEL_REPO_NAME)

exit_code = os.system('git reset --hard origin/master')
assert(exit_code == 0)

exit_code = os.system('git checkout -b %s' % args.branch)
assert(exit_code == 0)

exit_code = os.system('echo "-1" > version')
assert(exit_code == 0)

exit_code = os.system('git add version')
assert(exit_code == 0)

exit_code = os.system('git commit -m "Model branch = %s"' % args.branch)
assert(exit_code == 0)

exit_code = os.system('git push -u origin %s' % args.branch)
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


