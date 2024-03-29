import logging

from env_vars import *

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger('')

c_handler = logging.StreamHandler()
f_handler = logging.FileHandler(PROJECT_ROOT + '/indigo.log')
c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)

c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

logger.addHandler(c_handler)
logger.addHandler(f_handler)

