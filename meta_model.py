from torch import nn

import copy
import torch


class MetaModel(nn.Module):
    def __init__(self, config):
        super(MetaModel, self).__init__()

        self.learner = config['learner_model']
        self.learner_loss = config['learner_loss']
        self.learner_lr = config.get('learner_lr', 1e-3)

    def forward(self, support_data_loaders, query_data_loaders):
        query_outputs = []
        query_losses = []
        for support, query in zip(support_data_loaders, query_data_loaders):
            learner = copy.deepcopy(self.learner)
            learner.zero_grad()
            for batch_x, batch_y in support:
                output = learner(batch_x)
                loss = self.learner_loss(output, batch_y)
                grads = torch.autograd.grad(loss, learner.parameters())
                for param, grad in zip(learner.parameters(), grads):
                    param -= grad * self.learner_lr

            query_output = []
            query_loss = 0.0
            for batch_x, batch_y in query:
                output = learner(batch_x)
                loss = self.learner_loss(output, batch_y)
                loss.backward()
                query_loss += loss.item()
                query_output.extend(output.cpu().numpy())
            query_outputs.append(query_output)
            query_losses.append(query_loss)

            for param, new_param in zip(
                self.learner.parameters(), learner.parameters()
            ):
                if param.grad:
                    param.grad += new_param.grad
                else:
                    param.grad = new_param.grad
        return query_losses, query_outputs
