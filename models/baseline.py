import os

import torch

import coloredlogs
import logging

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
        self.stopping_threshold = config.get('stopping_threshold', 1e-2)

        if 'seq' in config['meta_model']:
            self.baseline_model = SeqBaselineModel(config)

        logger.info('Baseline instantiated')

    def training(self, train_episodes):
        best_loss = float('inf')
        patience = 0
        model_path = os.path.join(self.base_path, 'saved_models', 'Baseline-{}.h5'.format(self.stamp))
        for epoch in range(self.epochs):
            losses, accuracies = self.baseline_model(train_episodes, self.updates)
            avg_loss = sum(losses) / len(losses)
            avg_accuracy = sum(accuracies) / len(accuracies)
            logger.info('Epoch {}:\tavg loss={:.5f}\tavg accuracy={:.5f}'.format(
                epoch + 1, avg_loss, avg_accuracy
            ))
            if avg_loss < best_loss - self.stopping_threshold:
                patience = 0
                best_loss = avg_loss
                torch.save(self.baseline_model.learner.state_dict(), model_path)
                logger.info('Saving the model since the loss improved')
            else:
                patience += 1
                logger.info('Loss did not improve')
                if patience == self.early_stopping:
                    break
        self.baseline_model.learner.load_state_dict(torch.load(model_path))

    def testing(self, test_episodes):
        logger.info('---------- Baseline testing starts here ----------')
        for episode in test_episodes:
            prev_loss = 1e6
            patience = 0
            for epoch in range(self.epochs):
                logger.info('Meta epoch {}'.format(epoch + 1))
                loss = self.baseline_model([episode], self.updates, testing=True)
                loss = loss[0]
                if loss < prev_loss - self.stopping_threshold:
                    patience = 0
                    logger.info('Loss improved')
                else:
                    patience += 1
                    logger.info('Loss did not improve')
                    if patience == self.early_stopping:
                        break
                prev_loss = loss
