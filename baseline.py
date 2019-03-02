from abuse_baseline_model import AbuseBaseModel
from pos_baseline_model import POSBaseModel

import coloredlogs
import logging

logger = logging.getLogger('BaselineLog')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


class Baseline:
    def __init__(self, config):
        self.updates = config['num_updates']
        self.epochs = config['num_meta_epochs']
        if 'pos' in config['meta_model']:
            self.baseline_model = POSBaseModel(config)
        if 'abuse' in config['meta_model']:
            self.baseline_model = AbuseBaseModel(config)

    def training(self, support_loaders, query_loaders, languages):
        for support, query, language in zip(
            support_loaders, query_loaders, languages
        ):
            self.baseline_model(support, query, language, self.epochs)
            logger.info('')

    def testing(self, support_loaders, query_loaders, languages):
        logger.info('---------- Baseline testing starts here ----------')
        for support, query, lang in zip(
                support_loaders, query_loaders, languages
        ):
            self.baseline_model(support, query, lang, self.updates*10)
            if self.updates > 1:
                logger.info('')
