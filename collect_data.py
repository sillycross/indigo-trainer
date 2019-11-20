# collect a bunch of data points on a server
# This script MUST NOT be run concurrently!
#

import os
import argparse
import json
import threading
import os.path
from os import path
import time
import subprocess

from env_vars import *

parser = argparse.ArgumentParser()
parser.add_argument("--task", dest="task", help="the ordinal of task in config.json")
args = parser.parse_args()

assert(args.task != None)
args.task = int(args.task)

os.chdir(PROJECT_ROOT)

with open('workloads/config.json') as json_file:
	all_tasks = json.load(json_file)

assert(0 <= args.task and args.task < len(all_tasks))
task = all_tasks[args.task]

assert('num-clients' in task)
assert('client-duration' in task)
assert('mahimahi-command' in task)
assert('expert_cwnd' in task)

num_clients = int(task['num-clients'])
client_duration = int(task['client-duration'])
client_cmd = task['mahimahi-command']
expert_cwnd = task['expert_cwnd']

print('num_clients = %d, client_duration = %d' % (num_clients, client_duration))

# clear training output
#
exit_code = os.system("echo 'c' > /proc/indigo_training_output")
assert(exit_code == 0)

# rm the file that signals server exit
# we have to do this because it turns out that when iperf exit, 
# there may still be data in kernel buffer. And if the sh exits 
# while there are still data in the kernel buffer, it turns the iperf server 
# into a wierd state that must be killed by signal 9. This issue can be fixed 
# by making the sh run until all data in kernel buffer has been sent to server 
# and the server has exited 
#
os.system('rm server_exited')
assert(not path.exists('server_exited'))

def get_num_interfaces():
	cnt = 0
	with open('/proc/net/route') as fp:
		while (fp.readline()):
			cnt += 1
	return cnt

class run_shell(threading.Thread):
	def __init__(self, cmd):
		threading.Thread.__init__(self)
		self.cmd = cmd
		self.daemon = True
		self.exit_code = -1
		
	def run(self):
		print('Executing command: %s' % self.cmd)
		self.exit_code = os.system(self.cmd)

server = run_shell(cmd = 'iperf -s -p 10007 -P %d && touch server_exited' % num_clients)
server.start()
print('Server spawned!')

time.sleep(1)
	
# mahimahi has race in finding unused ports
# We have to wait until the previous mahimahi has binded the port
# before starting the next
#
num_interfaces = get_num_interfaces()

clients = []
for i in range(0, num_clients):
	clients.append(run_shell(cmd = '%s -- sh -c \'python2 run_client.py --duration %d --ca-name %s\'' % (client_cmd, client_duration, CONG_ALG_NAME)))
	clients[i].start()
	print('Client %d spawned!' % i)
	while (True):
		x = get_num_interfaces()
		if (x == num_interfaces + i + 1):
			break
		print('Getting %d interfaces expecting %d, waiting..' % (x, num_interfaces + i + 1))
		time.sleep(0.1)
		
server.join()
print('Server joined!')

for i in range(0, num_clients):
	clients[i].join()
	print('Client %d joined!' % i)

for i in range(0, num_clients):
	assert(clients[i].exit_code == 0)
	
assert(server.exit_code == 0)

# format training output into output file
#
exit_code = os.system('./training_output_formatter %d %d training_output.npz' % (EPISODE_LEN, expert_cwnd))
assert(exit_code == 0)

print('Done!')


