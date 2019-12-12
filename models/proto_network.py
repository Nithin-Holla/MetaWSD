import coloredlogs
import logging
import os
import torch

from models.seq_proto import SeqPrototypicalNetwork

logger = logging.getLogger('ProtoLearningLog')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


class PrototypicalNetwork:
    def __init__(self, config):
        self.base_path = config['base_path']
        self.stamp = config['stamp']
        self.updates = config['num_updates']
        self.meta_epochs = config['num_meta_epochs']
        self.early_stopping = config['early_stopping']
        self.stopping_threshold = config.get('stopping_threshold', 1e-2)

        if 'seq_meta' in config['meta_model']:
            self.proto_model = SeqPrototypicalNetwork(config)

        logger.info('Prototypical network instantiated')

    def training(self, train_episodes):
        best_loss = float('inf')
        patience = 0
        model_path = os.path.join(
            self.base_path, 'saved_models', 'ProtoNet-{}.h5'.format(self.stamp)
        )
        for epoch in range(self.meta_epochs):
            losses, accuracies = self.proto_model(train_episodes, self.updates)
            avg_loss = sum(losses) / len(losses)
            avg_accuracy = sum(accuracies) / len(accuracies)
            logger.info('Meta epoch {}:\tavg loss={:.5f}\tavg accuracy={:.5f}'.format(
                epoch + 1, avg_loss, avg_accuracy
            ))
            if avg_loss < best_loss - self.stopping_threshold:
                patience = 0
                best_loss = avg_loss
                torch.save(self.proto_model.learner.state_dict(), model_path)
                logger.info('Saving the model since the loss improved')
                logger.info('')
            else:
                patience += 1
                logger.info('Loss did not improve')
                logger.info('')
                if patience == self.early_stopping:
                    break
        self.proto_model.learner.load_state_dict(torch.load(model_path))

    def testing(self, test_episodes):
        logger.info('---------- Proto testing starts here ----------')
        for episode in test_episodes:
            for epoch in range(self.meta_epochs):
                logger.info('Meta epoch {}'.format(epoch + 1))
                self.proto_model([episode], self.updates, testing=True)