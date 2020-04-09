import coloredlogs
import logging
import torch
from allennlp.modules import Elmo
from allennlp.modules.elmo import batch_to_ids
from scipy.spatial.distance import cdist
from sklearn import metrics
import numpy as np

logger = logging.getLogger('NearestNeighbor Log')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


class NearestNeighborClassifier():
    def __init__(self, config):
        self.elmo = Elmo(options_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
                         weight_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
                         num_output_representations=1,
                         dropout=0,
                         requires_grad=False)
        self.device = torch.device(config.get('device', 'cpu'))
        self.elmo.to(self.device)
        logger.info('Nearest neighbor classifier instantiated')

    def vectorize(self, batch_x, batch_len, batch_y):
        with torch.no_grad():
            char_ids = batch_to_ids(batch_x)
            char_ids = char_ids.to(self.device)
            batch_x = self.elmo(char_ids)['elmo_representations'][0]
        batch_len = torch.tensor(batch_len).to(self.device)
        batch_y = torch.tensor(batch_y).to(self.device)
        return batch_x, batch_len, batch_y

    def training(self, train_episodes, val_episodes):
        return

    def testing(self, test_episodes):
        episode_accuracies, episode_precisions, episode_recalls, episodes_f1s = [], [], [], []
        for episode_id, episode in enumerate(test_episodes):
            batch_x, batch_len, batch_y = next(iter(episode.support_loader))
            support_repr, _, support_labels = self.vectorize(batch_x, batch_len, batch_y)
            support_repr = support_repr.view(support_repr.shape[0] * support_repr.shape[1], -1)
            support_labels = support_labels.view(-1)
            support_repr = support_repr[support_labels != -1].cpu().numpy()
            support_labels = support_labels[support_labels != -1].cpu().numpy()

            batch_x, batch_len, batch_y = next(iter(episode.query_loader))
            query_repr, _, true_labels = self.vectorize(batch_x, batch_len, batch_y)
            query_repr = query_repr.view(query_repr.shape[0] * query_repr.shape[1], -1)
            true_labels = true_labels.view(-1)
            query_repr = query_repr[true_labels != -1].cpu().numpy()
            true_labels = true_labels[true_labels != -1].cpu().numpy()

            dist = cdist(query_repr, support_repr, metric='cosine')
            nearest_neighbor = np.argmin(dist, axis=1)
            predictions = support_labels[nearest_neighbor]

            accuracy = metrics.accuracy_score(true_labels, predictions)
            precision = metrics.precision_score(true_labels, predictions, average='macro')
            recall = metrics.recall_score(true_labels, predictions, average='macro')
            f1_score = metrics.f1_score(true_labels, predictions, average='macro')
            logger.info('Episode {}/{}, task {} [query set]: Accuracy = {:.5f}, precision = {:.5f}, '
                        'recall = {:.5f}, F1 score = {:.5f}'.format(episode_id + 1, len(test_episodes), episode.task_id,
                                                                    accuracy, precision, recall, f1_score))

            episode_accuracies.append(accuracy)
            episode_precisions.append(precision)
            episode_recalls.append(recall)
            episodes_f1s.append(f1_score)

        logger.info('Avg meta-testing metrics: Accuracy = {:.5f}, precision = {:.5f}, recall = {:.5f}, '
                    'F1 score = {:.5f}'.format(np.mean(episode_accuracies),
                                               np.mean(episode_precisions),
                                               np.mean(episode_recalls),
                                               np.mean(episodes_f1s)))
