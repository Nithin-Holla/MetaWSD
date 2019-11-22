import torch
from sklearn import metrics


def calculate_metrics(predictions, labels, binary=False):
    averaging = 'binary' if binary else 'macro'
    predictions = torch.stack(predictions).cpu().numpy()
    labels = torch.stack(labels).cpu().numpy()
    accuracy = metrics.accuracy_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions, average=averaging)
    recall = metrics.recall_score(labels, predictions, average=averaging)
    f1_score = metrics.f1_score(labels, predictions, average=averaging)
    return accuracy, precision, recall, f1_score
