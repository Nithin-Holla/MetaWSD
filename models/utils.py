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


def make_prediction(output):
    with torch.no_grad():
        if output.size(1) == 1:
            pred = (torch.sigmoid(output) > 0.5).int()
        else:
            pred = output.max(-1)[1]
    return pred
