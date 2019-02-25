from base_models import RNNSequenceModel
from torch import nn
from torch import optim

import coloredlogs
import copy
import logging
import os
import torch

logger = logging.getLogger('POSBaselineLog')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


class POSBaseModel(nn.Module):
    def __init__(self, config):
        super(POSBaseModel, self).__init__()
        self.base = config['base']
        self.embeddings_file = config['embeddings']
        self.early_stopping = config['early_stopping']
        self.learner_loss = nn.CrossEntropyLoss()
        self.learner_lr = config.get('learner_lr', 1e-3)
        self.learner = RNNSequenceModel(config['learner_params'])
        if config.get('trained_baseline', None):
            self.learner.load_state_dict(torch.load(
                os.path.join(self.base, 'models', config['trained_baseline'])
            ))

    def forward(self, train_loader, test_loader, language, updates=1):
        self.learner.embedding.weight.data = self.load_embeddings(language)
        optimizer = optim.Adam(self.learner.parameters(), lr=self.learner_lr)

        best_loss = float('inf')
        best_model = None
        patience = 0
        for epoch in range(updates):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                output = self.learner(batch_x)
                loss = self.learner_loss(
                    output.view(output.size()[0] * output.size()[1], -1),
                    batch_y.view(-1)
                )
                loss.backward()
                optimizer.step()
            num_correct, num_total, total_loss = 0, 0, 0.0
            for batch_x, batch_y in test_loader:
                output = self.learner(batch_x)
                output = output.view(
                    output.size()[0] * output.size()[1], -1
                )
                batch_y = batch_y.view(-1)
                loss = self.learner_loss(output, batch_y)
                total_loss += loss.item()
                num_correct += torch.eq(
                    output.max(-1)[1], batch_y
                ).sum().item()
                num_total += batch_y.size()[0]
            logger.info('Language {}: loss = {:.5f} accuracy = {:.5f}'.format(
                language, total_loss, 1.0 * num_correct / num_total
            ))
            if total_loss < best_loss:
                patience = 0
                best_model = copy.deepcopy(self.learner)
            else:
                patience += 1
                if patience == self.early_stopping:
                    break
        self.learner = copy.deepcopy(best_model)

    def load_embeddings(self, language):
        file = os.path.join(
            self.base,
            self.embeddings_file.format(language=language)
        )
        embed_dim = self.learner.embed_dim
        embeds = [torch.zeros(embed_dim)]
        with open(file, 'r', encoding='utf-8') as vectors:
            count = 0
            for vector in vectors:
                count += 1
                if count == 1:
                    continue
                tokens = vector.strip().split()
                embed = [float(token) for token in tokens[-embed_dim:]]
                embeds.append(torch.Tensor(embed))
        embeds.append(self.learner.embedding.weight.data[-1])
        return torch.stack(embeds)
