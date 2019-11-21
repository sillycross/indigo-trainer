import os
import argparse
import json
import threading
import os.path
from os import path
import time
import subprocess
import socket
import datetime

from env_vars import *
from logger import logger

os.chdir(PROJECT_ROOT)

# repeatedly train the model until stop request is signaled
#

while True:
    if os.path.exists('stop_training'):
        logger.info('Stop signal received. Terminate.')
        break
        
    logger.info('%s: started new iteration...' % str(datetime.datetime.now()))
    exit_code = os.system('python3 run_one_iteration.py')
    assert(exit_code == 0)

