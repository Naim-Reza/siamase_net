import numpy as np
import matplotlib.pyplot as plt
import os
from utils import scatter_dict

LOG_DIR = './logs'
log_file = os.path.join(LOG_DIR, 'test_200_logs.npy')
test_logs = np.load(log_file, allow_pickle=True)

positive_distances, negative_distances = test_logs[0], test_logs[1]

x_p, y_p = scatter_dict(positive_distances)
x_n, y_n = scatter_dict(negative_distances)

plt.scatter(x_p, y_p)
plt.scatter(x_n, y_n)
plt.xlabel("Distance")
plt.ylabel("Classes")
plt.legend(['Positive Distance', 'Negative Distance'])
plt.show()
