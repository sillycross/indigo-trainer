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

os.chdir(PROJECT_ROOT + '/' + MODEL_REPO_NAME)

with open('version') as f:
	version_number = int(next(f))

if version_number == 0:
	commit_msg = "Initial model"
else:
	commit_msg = "Training iteration %d" % version_number
	
exit_code = os.system('git add .')
assert(exit_code == 0)

exit_code = os.system('git commit -m "%s"' % commit_msg)
assert(exit_code == 0)

exit_code = os.system('git push')
assert(exit_code == 0)

