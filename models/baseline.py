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

        if 'seq' in config['meta_model']:
            self.baseline_model = SeqBaselineModel(config)

        logger.info('Baseline instantiated')

    def training(self, train_episodes):
        self.baseline_model(train_episodes, self.epochs)

    def testing(self, test_episodes):
        logger.info('---------- Baseline testing starts here ----------')
        for episode in test_episodes:
            for epoch in range(self.meta_epochs):
                logger.info('Meta epoch {}'.format(epoch + 1))
                self.baseline_model([episode], self.updates)
