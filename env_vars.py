import os
import sys
from os import path

# information to access the model and LKM repo
#
GITHUB_USER_NAME = 'sillycross'
MODEL_REPO_NAME = 'indigo-trainer-checkpoint'
LKM_REPO_NAME = 'indigo-lkm'
TRAIN_REPO_NAME = 'indigo-trainer'

# are we using LSTM model or stateless model?
#
USING_LSTM_MODEL = True

PROJECT_ROOT = path.dirname(path.abspath(__file__))

# the name of the congestion control module in LKM
#
CONG_ALG_NAME = 'tcp_tron'

EPISODE_LEN = 1000

