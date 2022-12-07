import os
import argparse
from utils import show_training_results_from_file

parser = argparse.ArgumentParser(prog="LossGraph", description="Plot Loss Graph")
parser.add_argument('logdir')
parser.add_argument('logfile')

args = parser.parse_args()
LOG_DIR = args.logdir
LOG_FILE = os.path.join(LOG_DIR, args.logfile)

show_training_results_from_file(LOG_FILE)
