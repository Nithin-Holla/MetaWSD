import os

import torch

import coloredlogs
import logging
import numpy as np

from models.seq_baseline import SeqBaselineModel

logger = logging.getLogger('BaselineLog')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


class Baseline:
    def __init__(self, config):
        self.base_path = config['base_path']
        self.stamp = config['stamp']
        self.updates = config['num_updates']
        self.epochs = config['num_meta_epochs']
        self.early_stopping = config['early_stopping']
        self.stopping_threshold = config.get('stopping_threshold', 1e-3)

        if 'seq' in config['meta_model']:
            self.baseline_model = SeqBaselineModel(config)

        logger.info('Baseline instantiated')

    def training(self, train_episodes):
        best_loss = float('inf')
        best_f1 = 0
        patience = 0
        model_path = os.path.join(self.base_path, 'saved_models', 'Baseline-{}.h5'.format(self.stamp))
        logger.info('Model name: Baseline-{}.h5'.format(self.stamp))
        for epoch in range(self.epochs):
            losses, accuracies, precisions, recalls, f1s = self.baseline_model(train_episodes, self.updates)
            avg_loss = np.mean(losses)
            avg_accuracy = np.mean(accuracies)
            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)
            avg_f1 = np.mean(f1s)

            logger.info('Meta epoch {}: Avg loss = {:.5f}, avg accuracy = {:.5f}, avg precision = {:.5f}, '
                        'avg recall = {:.5f}, avg F1 score = {:.5f}'.format(epoch + 1, avg_loss, avg_accuracy,
                                                                            avg_precision, avg_recall, avg_f1))

            if avg_loss < best_loss - self.stopping_threshold:
                patience = 0
                best_loss = avg_loss
                best_f1 = avg_f1
                torch.save(self.baseline_model.learner.state_dict(), model_path)
                logger.info('Saving the model since the loss improved')
            else:
                patience += 1
                logger.info('Loss did not improve')
                if patience == self.early_stopping:
                    break
        self.baseline_model.learner.load_state_dict(torch.load(model_path))
        return best_f1

    def testing(self, test_episodes):
        logger.info('---------- Baseline testing starts here ----------')
        episode_accuracies, episode_precisions, episode_recalls, episodes_f1s = [], [], [], []
        for episode in test_episodes:
            best_loss = 1e6
            best_accuracy, best_precision, best_recall, best_f1_score = 0, 0, 0, 0
            patience = 0
            self.baseline_model.initialize_output_layer(episode.n_classes)
            for epoch in range(self.epochs):
                logger.info('Meta epoch {}'.format(epoch + 1))
                loss, accuracy, precision, recall, f1_score = self.baseline_model([episode], updates=1, testing=True)
                loss = loss[0]
                accuracy, precision, recall, f1_score = accuracy[0], precision[0], recall[0], f1_score[0]
                if loss < best_loss - self.stopping_threshold:
                    patience = 0
                    best_loss = loss
                    best_accuracy, best_precision, best_recall, best_f1_score = accuracy, precision, recall, f1_score
                    logger.info('Loss improved')
                else:
                    patience += 1
                    logger.info('Loss did not improve')
                    if patience == self.early_stopping:
                        break

            episode_accuracies.append(best_accuracy)
            episode_precisions.append(best_precision)
            episode_recalls.append(best_recall)
            episodes_f1s.append(best_f1_score)

        logger.info('Avg meta-testing metrics: Accuracy = {:.5f}, precision = {:.5f}, recall = {:.5f}, '
                    'F1 score = {:.5f}'.format(np.mean(episode_accuracies),
                                               np.mean(episode_precisions),
                                               np.mean(episode_recalls),
                                               np.mean(episodes_f1s)))
