import itertools
from collections import Counter

import coloredlogs
import logging
import torch
from sklearn import metrics
import numpy as np


logger = logging.getLogger('MajorityClassifier Log')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


class MajorityClassifier:
    def __init__(self):
        logger.info('Majority classifier instantiated')

    def training(self, train_episodes, val_episodes):
        return 0

    def testing(self, test_episodes):
        episode_accuracies, episode_precisions, episode_recalls, episodes_f1s = [], [], [], []

        for episode_id, episode in enumerate(test_episodes):
            for n_batch, (_, _, batch_y) in enumerate(episode.support_loader):
                pass
            class_counts = Counter(x for x in itertools.chain(*batch_y) if x != -1)
            majority_class = class_counts.most_common(1)[0][0]

            for n_batch, (_, _, batch_y) in enumerate(episode.query_loader):
                batch_y = torch.tensor(batch_y).view(-1)
                relevant_indices = torch.nonzero(batch_y != -1).view(-1).detach()
                true_labels = batch_y[relevant_indices]
                predictions = torch.full_like(true_labels, majority_class)

            true_labels = true_labels.numpy()
            predictions = predictions.numpy()

            accuracy = metrics.accuracy_score(true_labels, predictions)
            precision = metrics.precision_score(true_labels, predictions, average='macro')
            recall = metrics.recall_score(true_labels, predictions, average='macro')
            f1_score = metrics.f1_score(true_labels, predictions, average='macro')
            logger.info('Episode {}/{}, task {} [query set]: Accuracy = {:.5f}, precision = {:.5f}, '
                        'recall = {:.5f}, F1 score = {:.5f}'.format(episode_id + 1, len(test_episodes), episode.task_id,
                                                                    accuracy, precision, recall, f1_score))

            episode_accuracies.append(accuracy)
            episode_precisions.append(precision)
            episode_recalls.append(recall)
            episodes_f1s.append(f1_score)

        logger.info('Avg meta-testing metrics: Accuracy = {:.5f}, precision = {:.5f}, recall = {:.5f}, '
                    'F1 score = {:.5f}'.format(np.mean(episode_accuracies),
                                               np.mean(episode_precisions),
                                               np.mean(episode_recalls),
                                               np.mean(episodes_f1s)))
