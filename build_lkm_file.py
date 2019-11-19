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

# build the Linux Kernel Module from the compiled library

os.chdir(PROJECT_ROOT)

print('Copying model PB file to tensorflow..')

exit_code = os.system('cp %s/graph.pb tensorflow' % MODEL_REPO_NAME)
assert(exit_code == 0)

if USING_LSTM_MODEL:
	exit_code = os.system('cp graph.config.pbtxt tensorflow')
	assert(exit_code == 0)
else:
	exit_code = os.system('cp stateless_graph.config.pbtxt tensorflow/graph.config.pbtxt')
	assert(exit_code == 0)
os.chdir(PROJECT_ROOT + '/tensorflow')

# GCC sometimes crashes, retry on failure

print('Compiling model into object file..')

build_ok = False
num_retry = 0
while num_retry < 10:
	exit_code = os.system('bazel build --config=opt :graph')
	if exit_code == 0:
		build_ok = True
		break
	num_retry += 1
	print('Build fail, retrying.. (Attempt #%d)' % num_retry)
	
assert(build_ok)

# copy build artifact to model repo
 
print('Copying built model to model repo')
 
exit_code = os.system('mv bazel-genfiles/graph.h ../%s' % MODEL_REPO_NAME)
assert(exit_code == 0)

exit_code = os.system('mv bazel-genfiles/libgraph.pic.a ../%s' % MODEL_REPO_NAME)
assert(exit_code == 0)

exit_code = os.system('chmod 644 ../%s/libgraph.pic.a' % MODEL_REPO_NAME)
assert(exit_code == 0)

# copy model to LKM repo and build LKM

print('Building LKM')

os.chdir(PROJECT_ROOT)

exit_code = os.system('cp %s/graph.h %s/nn' % (MODEL_REPO_NAME, LKM_REPO_NAME))
assert(exit_code == 0)

exit_code = os.system('cp %s/libgraph.pic.a %s/nn' % (MODEL_REPO_NAME, LKM_REPO_NAME))
assert(exit_code == 0)

os.chdir(PROJECT_ROOT + '/' + LKM_REPO_NAME)

exit_code = os.system('make -B')
assert(exit_code == 0)

# copy build artifact to model repo

print('Copying built LKM to model repo')

os.chdir(PROJECT_ROOT)

exit_code = os.system('cp %s/indigo.ko %s' % (LKM_REPO_NAME, MODEL_REPO_NAME))
assert(exit_code == 0)

