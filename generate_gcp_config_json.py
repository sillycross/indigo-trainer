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

parser = argparse.ArgumentParser()
parser.add_argument("--num_leaves", dest="num_leaves", help="the total number of leaves")
parser.add_argument("--run_id", dest="run_id", help="the Run ID")
parser.add_argument("--zone", dest="zone", help="the zone of the instance group")
args = parser.parse_args()

assert(args.num_leaves != None)
args.num_leaves = int(args.num_leaves)
assert(args.num_leaves > 0)
assert(args.run_id != None)
assert(args.zone != None)

data = {
    'num_leaves': args.num_leaves,
	'run_id': args.run_id,
	'zone': args.zone
}

json_str = json.dumps(data, indent=4)
with open('gcp_config.json', 'w') as f:
    f.write(json_str)

