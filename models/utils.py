import copy
import numpy as np

import torch
from sklearn import metrics


def calculate_metrics(predictions, labels, binary=False):
    averaging = 'binary' if binary else 'macro'
    predictions = torch.stack(predictions).cpu().numpy()
    labels = torch.stack(labels).cpu().numpy()
    unique_labels = np.unique(labels)
    accuracy = metrics.accuracy_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions, average=averaging, labels=unique_labels)
    recall = metrics.recall_score(labels, predictions, average=averaging, labels=unique_labels)
    f1_score = metrics.f1_score(labels, predictions, average=averaging, labels=unique_labels)
    return accuracy, precision, recall, f1_score


def make_prediction(output):
    with torch.no_grad():
        if output.size(1) == 1:
            pred = (output > 0).int()
        else:
            pred = output.max(-1)[1]
    return pred


def replicate_model_to_gpus(model, device_ids):
    replica_models = [model] + [copy.deepcopy(model).to(device) for device in device_ids[1:]]
    for rm in replica_models[1:]:
        rm.device = next(rm.parameters()).device
    return replica_models
