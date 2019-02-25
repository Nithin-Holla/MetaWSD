from datetime import datetime
from meta_learning import MetaLearning
from torch.utils import data

import coloredlogs
import logging
import os
import pickle
import torch
torch.manual_seed(1025)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'stamp': str(datetime.now()).replace(':', '-').replace(' ', '_'),
    'learner_model': 'rnn_sequence',
    'learner_params': {
        'vocab_size': 200002,
        'hidden_size': 128,
        'num_outputs': 18,
        'embed_dim': 300,
    },
    'trained_learner': None,
    'num_shots': 10,
    'num_test_samples': 1500,
    'num_meta_epochs': 50,
    'early_stopping': 3,
    'data_files': os.path.join(
        'data_pos_tagging', 'dataset.{language}.pkl'
    ),
    'embeddings': os.path.join(
        'embeddings', 'wiki.multi.{language}.vec'
    ),
    'base': BASE_DIR,
}

POS_TAGS = {
    '<PAD>': 0, 'ADJ': 1, 'ADV': 2, 'INTJ': 3, 'NOUN': 4,
    'PROPN': 5, 'VERB': 6, 'ADP': 7, 'AUX': 8, 'CCONJ': 9,
    'DET': 10, 'NUM': 11, 'PART': 12, 'PRON': 13,
    'SCONJ': 14, 'PUNCT': 15, 'SYM': 16, 'X': 17,
}

logger = logging.getLogger('POSTaggingLog')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


class DataLoader(data.Dataset):
    def __init__(self, samples, classes, language):
        super(DataLoader, self).__init__()
        self.samples = samples
        self.classes = classes
        self.language = language

    def __getitem__(self, index):
        return self.samples[index], self.classes[index]

    def __len__(self):
        return len(self.classes)


def read_dataset(language):
    file = os.path.join(
        CONFIG['base'], CONFIG['data_files'].format(language=language)
    )
    with open(file, 'rb') as samples:
        return pickle.load(samples)


def load_vocab(language):
    vocab = {'<PAD>': 0}
    file = os.path.join(
        CONFIG['base'], CONFIG['embeddings'].format(language=language)
    )
    with open(file, 'r', encoding='utf-8') as vectors:
        count = 0
        for vector in vectors:
            count += 1
            if count > 1:
                word = vector[: vector.index(' ')]
                vocab[word] = len(vocab)
    vocab['<UNK>'] = len(vocab)
    return vocab


def produce_loader(language, samples, vocab):
    x, y = [], []
    max_len = 0
    total_samples = CONFIG['num_shots'] + CONFIG['num_test_samples']
    for tokens, tags in samples[:total_samples]:
        next_x, next_y = [], []
        for token, tag in zip(tokens, tags):
            if tag in POS_TAGS:
                next_x.append(vocab.get(token.lower(), vocab['<UNK>']))
                next_y.append(POS_TAGS[tag])
        x.append(next_x)
        y.append(next_y)
        max_len = max(max_len, len(next_y))
    for i in range(len(x)):
        while len(x[i]) < max_len:
            x[i].append(vocab['<PAD>'])
            y[i].append(POS_TAGS['<PAD>'])
    support = DataLoader(
        torch.LongTensor(x[:CONFIG['num_shots']]),
        torch.LongTensor(y[:CONFIG['num_shots']]),
        language
    )
    query = DataLoader(
        torch.LongTensor(x[-CONFIG['num_test_samples']:]),
        torch.LongTensor(y[-CONFIG['num_test_samples']:]),
        language
    )
    support_loader = data.DataLoader(support, batch_size=CONFIG['num_shots'])
    query_loader = data.DataLoader(query, batch_size=CONFIG['num_test_samples'])
    return support_loader, query_loader


if __name__ == "__main__":
    languages = ['bg', 'ca']
        # 'bg', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es',
        # 'et', 'fi', 'fr', 'he', 'hr', 'hu', 'id', 'it',
        # 'nl', 'no', 'pl', 'pt', 'ro', 'ru', 'sk',
        # 'sl', 'sv', 'tr', 'uk', 'vi'
    # ]

    support_loaders = []
    query_loaders = []
    for lang in languages:
        dataset = read_dataset(lang)
        vocabulary = load_vocab(lang)
        s, q = produce_loader(lang, dataset, vocabulary)
        support_loaders.append(s)
        query_loaders.append(q)
    logger.info('{} data loaders prepared'.format(len(languages)))
    meta_learner = MetaLearning(CONFIG)
    meta_learner.meta_training(support_loaders, query_loaders, languages)


