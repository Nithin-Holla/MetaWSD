from datetime import datetime
from torch.utils import data

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'stamp': str(datetime.now()).replace(':', '-').replace(' ', '_'),
    'learner_model': 'rnn_sequence',
    'learner_params': {
        'vocab_size': 200002,
        'hidden_size': 128,
        'num_outputs': 20,
        'embed_dim': 300,
    },
    'trained_learner': None,
    'num_shots': 10,
    'base': BASE_DIR,
}


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
    return language


def load_vocab_and_embeddings(language):
    return language, language


def produce_loader(language, vocab, samples):
    return vocab, samples, language


if __name__ == "__main__":
    pass
