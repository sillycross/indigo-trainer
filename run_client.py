import os
import argparse
import json
import threading
import os.path
from os import path
import time

parser = argparse.ArgumentParser()
parser.add_argument("--duration", dest="duration", help="the test duration")
parser.add_argument("--ca-name", dest="ca_name", help="the name of the congestion control algorithm")
args = parser.parse_args()
 
ca_name = args.ca_name
duration = int(args.duration)

cmd = "iperf -c $MAHIMAHI_BASE -p 10007 -Z %s -t %d" % (ca_name, duration)
print('Executing: %s' % cmd)
exit_code = os.system(cmd)
assert(exit_code == 0)
	
while (not path.exists('server_exited')):
	time.sleep(1)
	
print('Server seems to have quitted!')

