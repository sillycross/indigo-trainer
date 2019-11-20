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
parser.add_argument("--zone", dest="zone", default=gcp_zone_default, help="the Google Cloud Compute zone used to create instances")
parser.add_argument("--run-id", dest="run_id", help="The Run ID to tear down")
args = parser.parse_args()

assert(args.run_id != None)

os.chdir(PROJECT_ROOT)

instance_group_name = "indigo-%s" % args.run_id
exit_code = os.system('gcloud compute instance-groups list-instances %s --zone=%s > all_instances.txt' % (instance_group_name, args.zone))
assert(exit_code == 0)

with open('all_instances.txt') as f:
    content = f.read().splitlines()

instance_names = []
for i in range(1, len(content)):
    values = content[i].split()
    assert(len(values) == 3)
    assert(values[1] == args.zone)
    instance_names.append(values[0])

print('Found the following instances, which we will delete:')
print(instance_names)

s = ''
for instance_name in instance_names:
	s += ' ' + instance_name

print('Deleting instances, this may take a few minutes...')
cmd = "gcloud compute instances delete %s --zone=%s --quiet" % (s, args.zone)
exit_code = os.system(cmd)
assert(exit_code == 0)
print('All instances are deleted. Deleting instance group...')

cmd = "gcloud compute instance-groups unmanaged delete %s --zone=%s --quiet" % (instance_group_name, args.zone)
exit_code = os.system(cmd)
assert(exit_code == 0)
print('GCP environment teardown completed successfully.')

