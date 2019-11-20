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

os.chdir(PROJECT_ROOT)

exit_code = os.system('python3 install_pkgs.py')
assert(exit_code == 0)

exit_code = os.system('git clone https://github.com/%s/%s' % (GITHUB_USER_NAME, MODEL_REPO_NAME))
assert(exit_code == 0)

exit_code = os.system('git clone https://github.com/%s/%s' % (GITHUB_USER_NAME, LKM_REPO_NAME))
assert(exit_code == 0)

os.chdir(PROJECT_ROOT + '/' + LKM_REPO_NAME + '/training_output_formatter')

exit_code = os.system('make -B')
assert(exit_code == 0)

exit_code = os.system('cp training_output_formatter ../..')
assert(exit_code == 0)

print('Leaf setup complete.')

