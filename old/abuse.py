from old.baseline import Baseline
from datetime import datetime
from old.meta_learning import MetaLearning
from old.proto_learning import ProtoLearning
from torch.utils import data

import coloredlogs
import csv
import logging
import os
import torch
torch.manual_seed(1025)

logger = logging.getLogger('AbuseLog')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')

CONFIG = {
    'stamp': str(datetime.now()).replace(':', '-').replace(' ', '_'),
    'meta_model': 'abuse_meta_model',
    'learner_params': {
        'hidden_size': 128,
        'num_classes': 2,
        'embed_dim': 300,
        'kernel_sizes': [2, 3, 4],
        'num_filters': 300,
    },
    'trained_learner': None,
    'learner_lr': 1e-1,
    'meta_order': 1,
    'meta_lr': 1e-3,
    'meta_weight_decay': 1e-4,
    'num_shots': 10,
    'num_updates': 1,
    'num_test_samples': 500,
    'num_meta_epochs': 2,
    'early_stopping': 1,
    'data_files': os.path.join(
        'data_abuse', 'dataset.{identifier}.csv'
    ),
    'embeddings': os.path.join(
        'embeddings', 'glove.840B.300d.txt'
    ),
    'base': os.path.dirname(os.path.abspath(__file__)),
}
USED_SAMPLES = set()


class DataLoader(data.Dataset):
    def __init__(self, samples, classes, pad_value=0):
        super(DataLoader, self).__init__()
        lengths = [len(sample) for sample in samples]
        triplets = zip(samples, classes, lengths)
        triplets = sorted(triplets, key=lambda x: x[2], reverse=True)
        samples, classes, lengths = zip(*triplets)
        for i in range(len(lengths)):
            samples[i].extend([pad_value] * (lengths[0] - lengths[i]))
        self.samples = torch.LongTensor(samples)
        self.classes = torch.LongTensor(classes)
        self.lengths = torch.LongTensor(lengths)

    def __getitem__(self, index):
        return self.samples[index], self.classes[index], self.lengths[index]

    def __len__(self):
        return len(self.classes)


def tokenize_text(text):
    return text.split(' ')


def read_dataset(identifier, vocab):
    file = os.path.join(
        CONFIG['base'], CONFIG['data_files'].format(identifier=identifier)
    )
    with open(file, 'r', encoding='utf-8') as data_file:
        samples, classes = [], []
        dataset_reader = csv.reader(data_file)
        count = 0
        for line in dataset_reader:
            count += 1
            if count == 1:
                continue
            _, text, clazz = line
            classes.append(int(clazz))
            samples.append(
                [vocab.get(t, vocab['<UNK>']) for t in tokenize_text(text)]
            )
    return samples, classes


def produce_loaders(samples, classes, training=True):
    num_test_samples = CONFIG['num_test_samples']
    if training:
        num_test_samples = CONFIG['num_shots']
    x = [[], []]
    max_len = 250
    required_num = CONFIG['num_shots'] + num_test_samples
    for sample, clazz in zip(samples, classes):
        if len(sample) > max_len or len(x[clazz]) == required_num:
            continue
        hashed = hash(str(sample))
        if hashed in USED_SAMPLES:
            continue
        USED_SAMPLES.add(hashed)
        x[clazz].append(sample)
    support = DataLoader(
        x[0][:CONFIG['num_shots']] + x[1][:CONFIG['num_shots']],
        [0]*CONFIG['num_shots'] + [1]*CONFIG['num_shots']
    )
    query = DataLoader(
        x[0][CONFIG['num_shots']:] + x[1][CONFIG['num_shots']:],
        [0]*num_test_samples + [1]*num_test_samples
    )
    support_loader = data.DataLoader(support, batch_size=2*CONFIG['num_shots'])
    query_loader = data.DataLoader(query, batch_size=2*num_test_samples)
    return support_loader, query_loader


def load_vocab_and_embeddings():
    file = os.path.join(
        CONFIG['base'], CONFIG['embeddings']
    )
    embed_dim = CONFIG['learner_params']['embed_dim']
    embeds = [torch.zeros(embed_dim)]
    vocab = {'<PAD>': 0}
    with open(file, 'r', encoding='utf-8') as vectors:
        count = 0
        for vector in vectors:
            count += 1
            if count == 1:
                continue
            tokens = vector.strip().split()
            vocab[tokens[0]] = len(vocab)
            embed = [float(token) for token in tokens[-embed_dim:]]
            embeds.append(torch.Tensor(embed))
    embeds.append(torch.rand(embed_dim))
    vocab['<UNK>'] = len(vocab)
    CONFIG['embeddings'] = torch.stack(embeds)
    CONFIG['learner_params']['vocab_size'] = len(vocab)
    return vocab


if __name__ == "__main__":
    print(CONFIG)
    datasets = [
        'detox_attack', 'detox_toxicity', 'detox_attack', 'detox_toxicity',
        'detox_attack', 'detox_toxicity', 'detox_attack', 'detox_toxicity',
        'detox_attack', 'detox_toxicity', 'detox_attack', 'detox_toxicity',
        'detox_attack', 'detox_toxicity', 'detox_attack', 'detox_toxicity',
        'waseem_hovy'
    ]
    train_sets = {'detox_attack', 'detox_toxicity'}

    vocabulary = load_vocab_and_embeddings()
    support_loaders = []
    query_loaders = []
    for dataset in datasets:
        a, b = read_dataset(dataset, vocabulary)
        s, q = produce_loaders(a, b, dataset in train_sets)
        support_loaders.append(s)
        query_loaders.append(q)
    logger.info('{} data loaders prepared'.format(len(datasets)))

    train_supports, train_queries, train_datasets = [], [], []
    test_supports, test_queries, test_datasets = [], [], []
    for d in range(len(datasets)):
        if datasets[d] in train_sets:
            train_supports.append(support_loaders[d])
            train_queries.append(query_loaders[d])
            train_datasets.append(datasets[d])
        else:
            test_supports.append(support_loaders[d])
            test_queries.append(query_loaders[d])
            test_datasets.append(datasets[d])

    proto_learner = ProtoLearning(CONFIG)
    proto_learner.training(train_supports, train_queries, train_datasets)
    proto_learner.testing(test_supports, test_queries, test_datasets)

    abuse_base_model = Baseline(CONFIG)
    abuse_base_model.training(train_supports, train_queries, train_datasets)
    abuse_base_model.testing(test_supports, test_queries, test_datasets)

    meta_learner = MetaLearning(CONFIG)
    meta_learner.training(train_supports, train_queries, train_datasets)
    meta_learner.testing(test_supports, test_queries, test_datasets)
