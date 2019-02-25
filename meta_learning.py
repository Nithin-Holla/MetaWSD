from base_model import RNNSequenceModel
from meta_model import MetaModel
from torch import nn
from torch import optim

import coloredlogs
import logging
import os
import torch

logger = logging.getLogger('MetaLearningLog')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


class MetaLearning:
    def __init__(self, config):
        if 'rnn_sequence' in config['learner_model']:
            config['learner_loss'] = nn.CrossEntropyLoss()
            config['learner_model'] = RNNSequenceModel(
                config['learner_params']
            )

        self.meta_model = MetaModel(config)
        if config['trained_learner']:
            self.meta_model.learner.load_state_dict(torch.load(os.path.join(
                config['base'], 'models', config['trained_learner']
            )))
        self.base = config['base']
        self.stamp = config['stamp']
        self.meta_epochs = config['num_meta_epochs']
        self.early_stopping = config['early_stopping']

    def meta_training(self, support_loaders, query_loaders, languages):
        meta_optimizer = optim.Adam(self.meta_model.learner.parameters())

        best_loss = float('inf')
        patience = 0
        model_path = os.path.join(
            self.base, 'models', 'MetaModel-{}.h5'.format(self.stamp)
        )
        for epoch in range(self.meta_epochs):
            meta_optimizer.zero_grad()
            losses, accuracies = self.meta_model(
                support_loaders, query_loaders, languages
            )
            meta_optimizer.step()

            loss_value = torch.sum(torch.Tensor(losses)).item()
            accuracy = sum(accuracies) / len(accuracies)
            logger.info('Meta epoch {}:\tloss = {}\taccuracy = {}'.format(
                epoch + 1, loss_value, accuracy
            ))
            if loss_value <= best_loss:
                patience = 0
                best_loss = loss_value
                torch.save(self.meta_model.learner.state_dict(), model_path)
                logger.info('Saving the model since the loss improved')
                logger.info('')
            else:
                patience += 1
                logger.info('Loss did not improve')
                if patience == self.early_stopping:
                    break
                logger.info('')
        self.meta_model.learner.load_state_dict(torch.load(model_path))

    def meta_testing(self, support_loaders, query_loaders, languages):
        pass
