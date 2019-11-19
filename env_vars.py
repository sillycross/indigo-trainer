import os
import sys
from os import path

# information to access the model and LKM repo
#
GITHUB_USER_NAME = 'sillycross'
MODEL_REPO_NAME = 'indigo-trainer-checkpoint'
LKM_REPO_NAME = 'indigo-lkm'

# are we using LSTM model or stateless model?
#
USING_LSTM_MODEL = False

PROJECT_ROOT = path.dirname(path.abspath(__file__))

