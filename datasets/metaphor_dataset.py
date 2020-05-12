import csv
from collections import defaultdict

from torch.utils import data


class MetaphorDataset(data.Dataset):

    def __init__(self, data_path, n_classes):
        self.data_path = data_path
        self.sentences, self.labels = self._load_data()
        self.n_classes = n_classes

    def _load_data(self):
        sentences = []
        labels = []
        with open(self.data_path, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                sent = []
                lbl = []
                for token in row['sentence_txt'].split(' '):
                    if token.startswith('M_'):
                        lbl.append(1)
                        sent.append(token.split('M_')[1])
                    else:
                        lbl.append(0)
                        sent.append(token)
                sentences.append(sent)
                labels.append(lbl)
        return sentences, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]


class MetaphorClassificationDataset(data.Dataset):

    def __init__(self, data_path, n_classes):
        self.data_path = data_path
        self.sentences, self.verb_indices, self.verbs, self.labels = self._load_data()
        self.n_classes = n_classes
        # self.word_splits = self._split_by_word()

    def _load_data(self):
        sentences = []
        verb_indices = []
        verbs = []
        labels = []
        with open(self.data_path, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                sent = row['sentence'].split(' ')
                verb_idx = int(row['verb_idx'])
                verb = row['verb']
                lbl = [-1] * len(sent)
                lbl[verb_idx] = int(row['label'])
                sentences.append(sent)
                verb_indices.append(verb_idx)
                verbs.append(verb)
                labels.append(lbl)
        return sentences, verb_indices, verbs, labels

    # def _split_by_word(self):
    #     word_splits = defaultdict(lambda: defaultdict(list))
    #     for sent, verb_idx, verb, lbl in zip(self.sentences, self.verb_indices, self.verbs, self.labels):
    #         word_splits[verb]['sentences'].append(sent)
    #         label_vector = [-1] * len(sent)
    #         label_vector[verb_idx] = lbl
    #         word_splits[verb]['labels'].append(label_vector)
    #     return word_splits

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]


class WordMetaphorDataset(data.Dataset):
    def __init__(self, sentences, labels, n_classes):
        self.sentences = sentences
        self.labels = labels
        self.n_classes = n_classes

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]
