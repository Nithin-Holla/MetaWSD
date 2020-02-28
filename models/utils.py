import copy
import threading

import torch
from sklearn import metrics
from torch import nn
from torch._utils import ExceptionWrapper
from torch.nn import functional as F


def calculate_metrics(predictions, labels, binary=False):
    averaging = 'binary' if binary else 'macro'
    predictions = torch.stack(predictions).cpu().numpy()
    labels = torch.stack(labels).cpu().numpy()
    accuracy = metrics.accuracy_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions, average=averaging)
    recall = metrics.recall_score(labels, predictions, average=averaging)
    f1_score = metrics.f1_score(labels, predictions, average=averaging)
    return accuracy, precision, recall, f1_score


def make_prediction(output):
    with torch.no_grad():
        if output.size(1) == 1:
            pred = (output > 0).int()
        else:
            pred = output.max(-1)[1]
    return pred


def subset_softmax(output, unique_labels):
    new_output = torch.full_like(output, -45)
    new_output[:, unique_labels] = F.log_softmax(output[:, unique_labels], dim=1)
    return new_output


def replicate_model_to_gpus(model, device_ids):
    replica_models = [model] + [copy.deepcopy(model).to(device) for device in device_ids[1:]]
    for rm in replica_models[1:]:
        rm.device = next(rm.parameters()).device
    return replica_models
