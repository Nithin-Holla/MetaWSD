from abuse_proto_model import AbuseProtoModel

import coloredlogs
import logging

logger = logging.getLogger('MetaLearningLog')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


class ProtoLearning:
    def __init__(self, config):
        self.updates = config['num_updates']
        self.epochs = config['num_meta_epochs']
        if 'abuse_meta' in config['meta_model']:
            self.proto_model = AbuseProtoModel(config)
        logger.info('Proto learner instantiated')

    def training(self, support_loaders, query_loaders, identifiers):
        for support, query, idx in zip(
            support_loaders, query_loaders, identifiers
        ):
            self.proto_model(support, query, idx, self.epochs)
            logger.info('')

    def testing(self, support_loaders, query_loaders, identifiers):
        logger.info('---------- Proto testing starts here ----------')
        for support, query, idx in zip(
                support_loaders, query_loaders, identifiers
        ):
            self.proto_model(support, query, idx, self.updates+10)
