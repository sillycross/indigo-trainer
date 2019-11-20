import os
import argparse
import json
import threading
import os.path
from os import path
import time
import subprocess
import random
import numpy as np
import math

from env_vars import *

# generate the config file for all workloads 
#

NUM_SAMPLES_TO_COLLECT = 1000
NUM_SAMPLES_PER_RTT = 3

def get_trace_file_name(bw_mbps):
	return 'workloads/%dMbps.trace' % bw_mbps
	
def generate_trace(bw_mbps):
    # number of packets in 60 seconds
    num_packets = int(float(bw_mbps) * 5000)
    ts_list = np.linspace(0, 60000, num=num_packets, endpoint=False)

    # write timestamps to trace
    output_file = get_trace_file_name(bw_mbps)
    with open(output_file, 'w') as trace:
        for ts in ts_list:
            trace.write('%d\n' % ts)

def generate_simple_task(bw_mbps, delay_1w_ms, num_concurrent, num_repeats):
    global NUM_SAMPLES_TO_COLLECT, NUM_SAMPLES_PER_RTT
    
    trace_filename = get_trace_file_name(bw_mbps)
    mm_command = 'mm-delay %d mm-link %s %s' % (delay_1w_ms, trace_filename, trace_filename)
    
    samples_per_second = 1000.0 / 2 / delay_1w_ms * NUM_SAMPLES_PER_RTT
    mm_duration = int(round(1.0 * NUM_SAMPLES_TO_COLLECT / samples_per_second + 5))
    
    expert_cwnd = int(round(2 * delay_1w_ms * bw_mbps / 12.0))
    
    return {
        "mahimahi-command": mm_command,
		"client-duration": mm_duration,
		"expert_cwnd": expert_cwnd,
        "num-clients": num_concurrent,
        "repeats": num_repeats
    }

os.chdir(PROJECT_ROOT)
os.system('mkdir workloads')

all_tasks = []

# we make different numbers of connections to run concurrently to 
# make the environment more realistic (in real world there are >1 concurrent socket running)
# this is actually making an impact on the udp indigo: when there are multiple 
# sockets running, the udp indigo seems to converge to a lower cwnd than expert
# first value is # concurent sockets, second value is # repeats 
#
all_concurrency = [[1,4], [2,4], [4,2], [8,1]]

# part 1 training data:
# pairwise combination of a few BW/delay selections
#	
all_bw_mbps = [5, 10, 20, 30, 40, 50, 60]
all_delay_1w_ms = [10, 20, 40, 60, 80]

for bw in all_bw_mbps:
    generate_trace(bw)
    
for bw in all_bw_mbps:
    for delay in all_delay_1w_ms:
        for concurrency in all_concurrency:
            num_concurrent = concurrency[0] 
            num_repeats = concurrency[1]
            all_tasks.append(generate_simple_task(bw, delay, num_concurrent, num_repeats))

# part 2 training data:
# selection of different expert_cwnd, with random BW/delay to form that cwnd
#
all_expert_cwnd = []
for cwnd in range(10, 100, 10):
    all_expert_cwnd.append(cwnd)
for cwnd in range(100, 800, 20):
    all_expert_cwnd.append(cwnd)

for cwnd in all_expert_cwnd:
    for concurrency in all_concurrency:
        num_concurrent = concurrency[0] 
        num_repeats = concurrency[1]
        for i in range(0,num_repeats):
            # find a random bw/delay combination that results in the cwnd
            bdp = cwnd * 6
            if random.randint(0,1) == 0:
                # random on bw
                while True:
                    bw = random.randint(5, 60)
                    delay = bdp // bw
                    if delay >= 5 and delay <= 100:
                        break
            else:
                # random on delay
                while True:
                    delay = random.randint(10, 80)
                    bw = bdp // delay
                    if bw >= 5 and bw <= 70:
                        break
            generate_trace(bw)
            all_tasks.append(generate_simple_task(bw, delay, num_concurrent, 1))

# stats on total VM-seconds and # data points generated
#
total_cost = 0
total_data_points = 0
for task in all_tasks:
    total_cost += task['client-duration'] * task['repeats']
    total_data_points += task['num-clients'] * task['repeats']
print('Total cost is %d VM-seconds' % total_cost)
print('Total data point generated is %d' % total_data_points)

json_str = json.dumps(all_tasks, indent=4)
with open('workloads/config.json', 'w') as f:
    f.write(json_str)
    

