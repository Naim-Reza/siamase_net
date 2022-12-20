import matplotlib.pyplot as plt
import numpy as np
import glob
import os


def show_training_results(training_loss, validation_loss):
    plt.subplots(1, 3)
    plt.subplot(131), plt.xlabel('Epoch'), plt.ylabel('Loss')
    plt.plot(training_loss)
    plt.subplot(131)
    plt.plot(validation_loss)
    plt.subplot(132), plt.xlabel('Epoch'), plt.ylabel('Training Loss')
    plt.plot(training_loss)
    plt.subplot(133), plt.xlabel('Epoch'), plt.ylabel('Validation Loss')
    plt.plot(validation_loss)
    plt.show()


def show_training_results_from_file(file_path):
    logs = np.load(file_path)

    training_loss, validation_loss = logs[0], logs[1]

    show_training_results(training_loss, validation_loss)


def get_latest_weights(weight_dir):
    backbone_query = weight_dir + "/Backbone_*"
    head_query = weight_dir + "/Head_*"

    backbone_files = glob.glob(backbone_query)
    head_files = glob.glob(head_query)

    if not len(backbone_files) == 0 and not len(head_files) == 0:
        latest_backbone = max(backbone_files, key=os.path.getctime)
        latest_head = max(head_files, key=os.path.getctime)

        return latest_backbone, latest_head
    raise RuntimeError("No Backbone or Head weights found!!")


def to_array(dict):
    x = []
    y = []
    for key in dict.keys():
        _x = dict[key]
        _y = [key for _ in range(len(_x))]
        for i, item in enumerate(_y):
            x.append(_x[i])
            y.append(_y[i])

    return x, y
