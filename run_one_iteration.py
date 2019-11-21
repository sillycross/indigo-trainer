import os
import argparse
import json
import threading
import os.path
from os import path
import time
import subprocess
import socket
from queue import *
import random

from logger import logger
from env_vars import *

# run one iteration of training
#

os.chdir(PROJECT_ROOT)

with open('gcp_config.json') as json_file:
    data = json.load(json_file)
    assert('num_leaves' in data)
    NUM_LEAVES = data['num_leaves']
    assert('run_id' in data)
    RUN_ID = data['run_id']
    assert('zone' in data)
    GCP_ZONE = data['zone']

def ScpFromLeaf(leaf_id, remote_filename, local_filename):
    assert(0 <= leaf_id and leaf_id < NUM_LEAVES)
    retry_cnt = 0
    cmd = 'gcloud beta compute --project edgect-1155 scp indigo-%s-leaf%d:%s %s --zone %s' % (RUN_ID, leaf_id, remote_filename, local_filename, GCP_ZONE)
    while True:
        exit_code = os.system(cmd)
        if exit_code == 65280:
            # ssh failed with some small probablity
            # just retry in this case
            retry_cnt += 1
            logger.warning('***WARN*** ssh failed, retrying.. leaf_id = %d, cnt = %d' % (leaf_id, retry_cnt))
            if retry_cnt > 10:
                break
        else:
            break
    return exit_code

	
def ExecuteOnLeaf(leaf_id, command):
    assert(0 <= leaf_id and leaf_id < NUM_LEAVES)
    retry_cnt = 0
    cmd = 'gcloud beta compute --project edgect-1155 ssh --zone %s indigo-%s-leaf%d -- "%s"' % (GCP_ZONE, RUN_ID, leaf_id, command)
    while True:
        exit_code = os.system(cmd)
        if exit_code == 65280:
            # ssh failed with some small probablity
            # just retry in this case
            retry_cnt += 1
            logger.warning('***WARN*** ssh failed, retrying.. leaf_id = %d, cnt = %d' % (leaf_id, retry_cnt))
            if retry_cnt > 10:
                break
        else:
            break
    return exit_code

class AsyncRunOnLeaf(threading.Thread):
    def __init__(self, leaf_id, fn):
        threading.Thread.__init__(self)
        self.fn = fn
        self.leaf_id = leaf_id
        self.daemon = True
        self.exit_code = -1
		
    def run(self):
        self.exit_code = self.fn(self.leaf_id)
	
def ExecuteOnAllLeaves(fn):
    threads = []
    for i in range(0, NUM_LEAVES):
        threads.append(AsyncRunOnLeaf(i, fn))
        threads[i].start()
	
    for i in range(0, NUM_LEAVES):
        threads[i].join()
	
    for i in range(0, NUM_LEAVES):
        assert(threads[i].exit_code == 0)
		
os.chdir(PROJECT_ROOT + '/' + MODEL_REPO_NAME)

# discard all changes from maybe failed previous iteration
#
exit_code = os.system('git reset --hard')
assert(exit_code == 0) 

# read version number
#
with open('version') as f:
    values = f.read().splitlines()
    assert(len(values) == 1)
    VERSION = int(values[0])
    assert(VERSION >= 0)

os.chdir(PROJECT_ROOT)

logger.info('****** Training on model version %d ******' % VERSION)

def leaf_init(leaf_id):
    return ExecuteOnLeaf(leaf_id, 'cd %s/%s && git pull && git checkout %s && [ \'0\' == \\"$(cat %s/version)\\" ] && sudo insmod indigo.ko' % (TRAIN_REPO_NAME, MODEL_REPO_NAME, RUN_ID, MODEL_REPO_NAME))

def leaf_update(leaf_id):
    return ExecuteOnLeaf(leaf_id, 'cd %s/%s && sudo rmmod indigo.ko && git pull && && [ \'%d\' == \\"$(cat %s/version)\\" ] && sudo insmod indigo.ko' % (TRAIN_REPO_NAME, MODEL_REPO_NAME, VERSION, MODEL_REPO_NAME))
	
if (VERSION == 0):
    # for the first iteration, let the leaves pull the repo, checkout correct branch, and insmod
    ExecuteOnAllLeaves(leaf_init)
else:
    # for later iterations, just rmmod, pull the repo, and insmod
    ExecuteOnAllLeaves(leaf_update)
	
logger.info('****** All leaves updated LKM successfully, distributing tasks to leaves ******')

# OK if exists
os.system('mkdir training_data')

# OK if not exist
os.system('rm -rf training_data/%d' % VERSION)

exit_code = os.system('mkdir training_data/%d' % VERSION)
assert(exit_code == 0)

q = Queue()

def collect_sample(leaf_id, task):
    task_id = task[0]
    repeat_id = task[1]
    exit_code = ExecuteOnLeaf(leaf_id, 'cd %s && python3 collect_data.py --task %d' % (TRAIN_REPO_NAME, task_id))
    if (exit_code != 0):
        logger.error('***ERR*** collect_data.py failed with exit code %d' % exit_code)
        return exit_code
    
    exit_code = ScpFromLeaf(leaf_id, '%s/training_output.npz' % TRAIN_REPO_NAME, 'training_data/%d/%d_%d.npz' % (VERSION, task_id, repeat_id))
    if (exit_code != 0):
        logger.error('***ERR*** scp failed with exit code %d' % exit_code)
        return exit_code
    
    return 0
    
def leaf_fn(leaf_id):
    ret = 0
    while True:
        item = q.get()
        if item is None:
            break
        exit_code = collect_sample(leaf_id, item)
        if (exit_code != 0):
            logger.error('***ERR*** Command failed with exit code %d! Leaf id: %d, item %s' % (exit_code, leaf_id, str(item)))
            ret = exit_code
        q.task_done()
    return ret
    
with open('workloads/config.json') as json_file:
    tasks_json = json.load(json_file) 

all_tasks = []
for i in range(0, len(tasks_json)):
    task = tasks_json[i]
    assert('repeats' in task)
    repeats = task['repeats']
    for k in range(0, repeats):
        all_tasks.append([i, k])
        
random.shuffle(all_tasks)

threads = []
for i in range(0, NUM_LEAVES):
    threads.append(AsyncRunOnLeaf(i, leaf_fn))
    threads[i].start()

for item in all_tasks:
    q.put(item)

q.join()

for i in range(0, NUM_LEAVES):
    q.put(None)

for i in range(0, NUM_LEAVES):
    threads[i].join()

for i in range(0, NUM_LEAVES):
    assert(threads[i].exit_code == 0)
    
logger.info('****** All leaf tasks completed successfully, training model ******')

start_time = time.time()

exit_code = os.system('python3 run_trainer.py')
assert(exit_code == 0)

end_time = time.time()

logger.info('****** Training complete in %f sec, building .pb file ******' % (float(end_time - start_time)))

exit_code = os.system('python3 build_pb_file.py')
assert(exit_code == 0)

logger.info('****** .pb file build complete, building LKM file ******')

exit_code = os.system('python3 build_lkm_file.py')
assert(exit_code == 0)

logger.info('****** LKM file build complete, incrementing model version to %d and committing model ******' % (VERSION + 1))

exit_code = os.system('echo "%d" > %s/version' % (VERSION + 1, MODEL_REPO_NAME))
assert(exit_code == 0)

exit_code = os.system('python3 commit_model.py')
assert(exit_code == 0)

logger.info('****** Training on model version %d complete, now model version is incremented to %d ******' % (VERSION, VERSION + 1))

