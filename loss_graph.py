import os
from utils import show_training_results_from_file

LOG_DIR = './logs'
LOG_FILE = os.path.join(LOG_DIR, 'train_log.npy')

show_training_results_from_file(LOG_FILE)
