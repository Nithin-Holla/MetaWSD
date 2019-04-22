from base_models import RNNSequenceModel
from torch import nn

import coloredlogs
import copy
import logging
import os
import torch

logger = logging.getLogger('Log')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


class POSMetaModel(nn.Module):
    def __init__(self, config):
        super(POSMetaModel, self).__init__()
        self.base = config['base']
        self.embeddings_file = config['embeddings']
        self.learner_loss = nn.CrossEntropyLoss()
        self.learner_lr = config.get('learner_lr', 1e-3)
        self.learner = RNNSequenceModel(config['learner_params'])
        if config['trained_learner']:
            self.learner.load_state_dict(torch.load(
                os.path.join(self.base, 'models', config['trained_learner'])
            ))

    def forward(self, support_loaders, query_loaders, languages, updates=1):
        query_losses = []
        accuracies = []
        for support, query, lang in zip(
                support_loaders, query_loaders, languages
        ):
            learner = copy.deepcopy(self.learner)
            learner.embedding.weight.data = self.load_embeddings(lang)
            num_correct, num_total, query_loss = 0, 0, 0.0
            for _ in range(updates):
                for batch_x, batch_y in support:
                    output = learner(batch_x)
                    loss = self.learner_loss(
                        output.view(output.size()[0] * output.size()[1], -1),
                        batch_y.view(-1)
                    )
                    params = [
                        p for p in learner.parameters() if p.requires_grad
                    ]
                    grads = torch.autograd.grad(loss, params)
                    for param, grad in zip(params, grads):
                        param.data -= grad * self.learner_lr

                num_correct, num_total, query_loss = 0, 0, 0.0
                learner.zero_grad()
                for batch_x, batch_y in query:
                    output = learner(batch_x)
                    output = output.view(
                        output.size()[0] * output.size()[1], -1
                    )
                    batch_y = batch_y.view(-1)
                    loss = self.learner_loss(output, batch_y)
                    loss.backward()
                    query_loss += loss.item()
                    num_correct += torch.eq(
                        output.max(-1)[1], batch_y
                    ).sum().item()
                    num_total += batch_y.size()[0]
                logger.info('Language {}: loss = {:.5f} accuracy = {:.5f}'.format(
                    lang, query_loss, 1.0 * num_correct / num_total
                ))
            query_losses.append(query_loss)
            accuracies.append(1.0 * num_correct / num_total)

            for param, new_param in zip(
                self.learner.parameters(), learner.parameters()
            ):
                if param.grad is not None and param.requires_grad:
                    param.grad += new_param.grad
                elif param.requires_grad:
                    param.grad = new_param.grad
        # Average the accumulated gradients
        for param in self.learner.parameters():
            if param.requires_grad:
                param.grad /= len(accuracies)
        return query_losses, accuracies

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
