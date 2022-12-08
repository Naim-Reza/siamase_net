import os
import argparse
import numpy as np
from utils import scatter_dict
import csv


def get_distances(log_file_path):
    test_logs = np.load(log_file_path, allow_pickle=True)

    positive_distances, negative_distances = test_logs[0], test_logs[1]

    x_p, y_p = scatter_dict(positive_distances)
    x_n, y_n = scatter_dict(negative_distances)

    return x_p, x_n


LOG_DIR = './logs'
SIAMASE_DIR = os.path.join(LOG_DIR, 'test_siamese_200_1_logs.npy')
SOFTMAX_DIR = os.path.join(LOG_DIR, 'test_softmax_200_1_logs.npy')
SPHERE_DIR = os.path.join(LOG_DIR, 'test_sphere_200_1_logs.npy')
COSFACE_DIR = os.path.join(LOG_DIR, 'test_cosface_200_1_logs.npy')
ARCFACE_DIR = os.path.join(LOG_DIR, 'test_arcface_200_1_logs.npy')

dirs = [SIAMASE_DIR, SOFTMAX_DIR, SPHERE_DIR, COSFACE_DIR, ARCFACE_DIR]

parser = argparse.ArgumentParser(prog="generate_result_csv",
                                 description='Reads numpy log files from LOG_DIR and generates csv with the mean values of positive and negative loss for different head models. ')

parser.add_argument('outputFileName')
args = parser.parse_args()

# siamese_pd, siamese_nd = get_distances(SIAMASE_DIR)
# softmax_pd, softmax_nd = get_distances(SOFTMAX_DIR)
# sphere_pd, sphere_nd = get_distances(SPHERE_DIR)
# cosface_pd, cosface_nd = get_distances(COSFACE_DIR)
# arcface_pd, arcface_nd = get_distances(ARCFACE_DIR)
#
# siamese_pd_mean, siamese_nd_mean = np.mean(siamese_pd), np.mean(siamese_nd)
# softmax_pd_mean, softmax_nd_mean = np.mean(softmax_pd), np.mean(softmax_nd)
# sphere_pd_mean, sphere_nd_mean = np.mean(sphere_pd), np.mean(sphere_nd)
# cosface_pd_mean, cosface_nd_mean = np.mean(cosface_pd), np.mean(cosface_nd)
# arcface_pd_mean, arcface_nd_mean = np.mean(arcface_pd), np.mean(arcface_nd)

headers = ['Head Name', 'Average Positive Distance', 'Average Negative Distance']
head_names = ['Siamese', 'SoftMax', 'SphereFace', 'CosFace', 'ArcFace']
output_file = os.path.join(LOG_DIR, args.outputFileName)

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    for i, path in enumerate(dirs):
        pd, nd = get_distances(path)
        pd_mean, nd_mean = np.mean(pd), np.mean(nd)
        head = head_names[i]
        row = [head, pd_mean, nd_mean]
        writer.writerow(row)
