from abuse_cnn_meta_model import AbuseCNNMetaModel
from abuse_meta_model import AbuseMetaModel
from torch import optim

import coloredlogs
import logging
import os
import torch

from models.seq_meta import SeqMetaModel

logger = logging.getLogger('MetaLearningLog')
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

        if 'seq_meta' in config['meta_model']:
            self.meta_model = SeqMetaModel(config)
        if 'abuse_meta' in config['meta_model']:
            self.meta_model = AbuseMetaModel(config)
        if 'abuse_cnn_meta' in config['meta_model']:
            self.meta_model = AbuseCNNMetaModel(config)

        logger.info('Meta learner instantiated')

    def training(self, train_episodes):
        meta_optimizer = optim.Adam(
            self.meta_model.learner.parameters(), lr=self.meta_lr,
            weight_decay=self.meta_weight_decay
        )
        best_loss = float('inf')
        patience = 0
        model_path = os.path.join(
            self.base_path, 'saved_models', 'MetaModel-{}.h5'.format(self.stamp)
        )
        for epoch in range(self.meta_epochs):
            meta_optimizer.zero_grad()
            losses, accuracies = self.meta_model(train_episodes, self.updates)
            meta_optimizer.step()

            loss_value = torch.mean(torch.Tensor(losses)).item()
            accuracy = sum(accuracies) / len(accuracies)
            logger.info('Meta epoch {}:\tavg loss={:.5f}\tavg accuracy={:.5f}'.format(
                epoch + 1, loss_value, accuracy
            ))
            if loss_value <= best_loss:
                patience = 0
                best_loss = loss_value
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
            self.meta_model([episode], self.updates+10)
            logger.info('')
