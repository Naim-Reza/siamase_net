import numpy as np
import matplotlib.pyplot as plt
import os
from utils import to_array
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test_graph', description='Plot Scatter graph')
    parser.add_argument('logFileName')

    args = parser.parse_args()

    LOG_DIR = './logs'
    log_file = os.path.join(LOG_DIR, args.logFileName)
    test_logs = np.load(log_file, allow_pickle=True)

    positive_distances, negative_distances = test_logs[0], test_logs[1]

    x_p, y_p = to_array(positive_distances)
    x_n, y_n = to_array(negative_distances)
    print(y_p)

    plt.scatter(x_p, y_p, alpha=0.5)
    plt.scatter(x_n, y_n, alpha=0.5)
    plt.xlabel("Distance")
    plt.ylabel("Classes")
    plt.legend(['Positive Distance', 'Negative Distance'])
    plt.show()
