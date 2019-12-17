from torch import optim

import coloredlogs
import logging
import os
import torch

from models.seq_meta import SeqMetaModel

logger = logging.getLogger('MAML Log')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


class MAML:
    def __init__(self, config):
        self.base_path = config['base_path']
        self.stamp = config['stamp']
        self.updates = config['num_updates']
        self.meta_epochs = config['num_meta_epochs']
        self.early_stopping = config['early_stopping']
        self.meta_lr = config.get('meta_lr', 1e-3)
        self.meta_weight_decay = config.get('meta_weight_decay', 0.0)
        self.stopping_threshold = config.get('stopping_threshold', 1e-2)

        if 'seq_meta' in config['meta_model']:
            self.meta_model = SeqMetaModel(config)

        logger.info('MAML instantiated')

    def training(self, train_episodes):
        learner_params = [p for p in self.meta_model.learner.parameters() if p.requires_grad]
        meta_optimizer = optim.Adam(learner_params, lr=self.meta_lr, weight_decay=self.meta_weight_decay)
        best_loss = float('inf')
        patience = 0
        model_path = os.path.join(self.base_path, 'saved_models', 'MetaModel-{}.h5'.format(self.stamp))
        for epoch in range(self.meta_epochs):
            meta_optimizer.zero_grad()
            losses, accuracies = self.meta_model(train_episodes, self.updates)
            meta_optimizer.step()

            avg_loss = sum(losses) / len(losses)
            avg_accuracy = sum(accuracies) / len(accuracies)
            logger.info('Meta epoch {}:\tavg loss={:.5f}\tavg accuracy={:.5f}'.format(
                epoch + 1, avg_loss, avg_accuracy
            ))
            if avg_loss < best_loss - self.stopping_threshold:
                patience = 0
                best_loss = avg_loss
                torch.save(self.meta_model.learner.state_dict(), model_path)
                logger.info('Saving the model since the loss improved')
            else:
                patience += 1
                logger.info('Loss did not improve')
                if patience == self.early_stopping:
                    break
        self.meta_model.learner.load_state_dict(torch.load(model_path))

    def testing(self, test_episodes):
        logger.info('---------- Meta testing starts here ----------')
        for episode in test_episodes:
            prev_loss = 1e6
            patience = 0
            self.meta_model.output_layer[episode.task].reset_parameters()
            for epoch in range(self.meta_epochs):
                logger.info('Meta epoch {}'.format(epoch + 1))
                loss = self.meta_model([episode], self.updates, testing=True)
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
