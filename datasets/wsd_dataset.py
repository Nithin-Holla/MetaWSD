import itertools
import json
import os
from collections import defaultdict

from torch.utils import data
import xml.etree.ElementTree as ET
import numpy as np

from datasets import utils


class SemCorWSDDataset(data.Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.sense_inventory = self._load_sense_inventory()
        self.sentences, self.lemmatized_sentences, self.labels = self._load_data()
        self.word_splits = self._split_by_word()

    def _load_sense_inventory(self):
        sense_inventory_file = os.path.join(self.data_path, 'semcor_sense_inventory.json')
        with open(sense_inventory_file, 'r', encoding='utf8') as f:
            sense_inventory = json.load(f)
        return sense_inventory

    def _load_data(self):
        sentences = []
        lemmatized_sentences = []
        labels = []
        for file_name in os.listdir(self.data_path):
            if file_name.endswith('.xml'):
                sent = []
                lem_sent = []
                lbl = []
                tree = ET.parse(os.path.join(self.data_path, file_name))
                root = tree.getroot()
                for child in root:
                    token = child.attrib['text']
                    lemma = child.attrib['lemma'].lower() if 'lemma' in child.attrib else token.lower()
                    break_level = child.attrib['break_level']
                    if 'sense' in child.attrib and lemma in self.sense_inventory:
                        sense_lbl = self.sense_inventory[lemma].index(child.attrib['sense'])
                    else:
                        sense_lbl = -1
                    if break_level == 'PARAGRAPH_BREAK' or break_level == 'SENTENCE_BREAK':
                        sentences.append(sent)
                        lemmatized_sentences.append(lem_sent)
                        labels.append(lbl)
                        sent = [token]
                        lem_sent = [lemma]
                        lbl = [sense_lbl]
                    else:
                        sent.append(token)
                        lem_sent.append(lemma)
                        lbl.append(sense_lbl)
                sentences.append(sent)
                lemmatized_sentences.append(lem_sent)
                labels.append(lbl)
        return sentences, lemmatized_sentences, labels

    def _split_by_word(self):
        word_splits = defaultdict(lambda: defaultdict(list))
        for sent, lem_sent, lbl in zip(self.sentences, self.lemmatized_sentences, self.labels):
            for word, lemma, sense_lbl in zip(sent, lem_sent, lbl):
                if lemma in self.sense_inventory and sense_lbl != -1:
                    matching_indices = [i for i, (x, y) in enumerate(zip(lem_sent, lbl)) if x == lemma and y != -1]
                    if len(matching_indices) != 0:
                        modified_lbl = [lbl[i] if i in matching_indices else -1 for i in range(len(lbl))]
                        word_splits[lemma]['sentences'].append(sent)
                        word_splits[lemma]['labels'].append(modified_lbl)
        return word_splits

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]


class WordWSDDataset(data.Dataset):

    def __init__(self, sentences, labels, n_classes):
        self.sentences = sentences
        self.labels = labels
        self.n_classes = n_classes

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]


class MetaWSDDataset(data.Dataset):

    def __init__(self, file_name):
        json_dict = utils.read_json(file_name)
        self.sentences, self.labels = [],  []
        for entry in json_dict:
            self.sentences.append(entry['sentence'])
            self.labels.append(entry['label'])
        self.n_classes = np.max(list(itertools.chain(*self.labels))) + 1

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]
