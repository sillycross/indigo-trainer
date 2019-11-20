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

# additional packages & configurations that we forgot to put into the initial image
#

exit_code = os.system('sudo apt-get update')
assert(exit_code == 0)

exit_code = os.system('sudo apt-get install -y libz-dev iperf mahimahi')
assert(exit_code == 0)

exit_code = os.system('sudo sysctl -w net.ipv4.ip_forward=1')
assert(exit_code == 0)

