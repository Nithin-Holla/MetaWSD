from torch import nn

import copy
import os
import torch


class MetaModel(nn.Module):
    def __init__(self, config):
        super(MetaModel, self).__init__()
        self.base = config['base']
        self.embeddings_file = config['embeddings']
        self.learner = config['learner_model']
        self.learner_loss = config['learner_loss']
        self.learner_lr = config.get('learner_lr', 1e-3)

    def forward(self, support_loaders, query_loaders, languages):
        query_outputs = []
        query_losses = []
        for support, query, lang in zip(
                support_loaders, query_loaders, languages
        ):
            learner = copy.deepcopy(self.learner)
            learner.embedding.weight.data = self.load_embeddings(lang)
            learner.zero_grad()
            for batch_x, batch_y in support:
                output = learner(batch_x)
                loss = self.learner_loss(
                    output.view(output.size()[0] * output.size()[1], -1),
                    batch_y.view(-1)
                )
                params = [p for p in learner.parameters() if p.requires_grad]
                grads = torch.autograd.grad(loss, params)
                for param, grad in zip(params, grads):
                    param.data -= grad * self.learner_lr

            query_output = []
            query_loss = 0.0
            for batch_x, batch_y in query:
                output = learner(batch_x)
                loss = self.learner_loss(
                    output.view(output.size()[0] * output.size()[1], -1),
                    batch_y.view(-1)
                )
                loss.backward()
                query_loss += loss.item()
                query_output.extend(output.data.numpy())
            query_outputs.append(query_output)
            query_losses.append(query_loss)
            print('Loss on language {} is {}'.format(lang, query_loss))

            for param, new_param in zip(
                self.learner.parameters(), learner.parameters()
            ):
                if param.grad is not None and param.requires_grad:
                    param.grad += new_param.grad
                elif param.requires_grad:
                    param.grad = new_param.grad
        return query_losses, query_outputs

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
