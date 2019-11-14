from abuse_proto_model import AbuseProtoModel

import coloredlogs
import logging
import os
import torch

logger = logging.getLogger('ProtoLearningLog')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


class ProtoLearning:
    def __init__(self, config):
        self.base = config['base']
        self.stamp = config['stamp']
        self.updates = config['num_updates']
        self.meta_epochs = config['num_meta_epochs']
        self.early_stopping = config['early_stopping']
        if 'abuse_meta' in config['meta_model']:
            self.proto_model = AbuseProtoModel(config)
        logger.info('Prototypical encoder instantiated')

    def training(self, support_loaders, query_loaders, identifiers):
        best_loss = float('inf')
        patience = 0
        model_path = os.path.join(
            self.base, 'saved_models', 'ProtoModel-{}.h5'.format(self.stamp)
        )
        for epoch in range(self.meta_epochs):
            losses, accuracies = self.proto_model(
                support_loaders, query_loaders, identifiers, self.updates
            )
            loss_value = torch.mean(torch.Tensor(losses)).item()
            accuracy = sum(accuracies) / len(accuracies)
            logger.info('Meta epoch {}:\tavg loss={:.5f}\tavg accuracy={:.5f}'.format(
                epoch + 1, loss_value, accuracy
            ))
            if loss_value <= best_loss:
                patience = 0
                best_loss = loss_value
                torch.save(self.proto_model.encoder.state_dict(), model_path)
                logger.info('Saving the model since the loss improved')
                logger.info('')
            else:
                patience += 1
                logger.info('Loss did not improve')
                logger.info('')
                if patience == self.early_stopping:
                    break
        self.proto_model.encoder.load_state_dict(torch.load(model_path))

    def testing(self, support_loaders, query_loaders, identifiers):
        logger.info('---------- Proto testing starts here ----------')
        for support, query, idx in zip(
                support_loaders, query_loaders, identifiers
        ):
            self.proto_model([support], [query], [idx], self.updates)
            logger.info('')
