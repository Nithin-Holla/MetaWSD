from torch import nn
from torch.optim import Adam


class MetaModel(nn.Module):
    def __init__(self, config):
        super(MetaModel, self).__init__()

        self.learner = config['learner_model']
        self.learner_loss = config['learner_loss']
        self.learner_lr = config.get('learner_lr', 1e-3)

        self.num_shots = config.get('num_shots', 5)
        self.meta_epochs = config.get('meta_epochs', 50)
        self.early_stopping = config.get('early_stopping', 3)
        self.meta_lr = config.get('meta_lr', 1e-3)
        self.meta_optimizer = Adam(self.learner.parameters(), lr=self.meta_lr)

    def forward(self, support_data_loaders, query_data_loaders):
        query_outputs = []
        query_losses = []

        for k in range(self.num_shots + 1):
            pass

        return query_outputs, query_losses
