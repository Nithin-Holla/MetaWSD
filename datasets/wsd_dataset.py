import json
import os

from torch.utils import data
import xml.etree.ElementTree as ET

from datasets import utils


class WSDDataset(data.Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.sense_vocab = self._load_sense_vocab()
        self.n_classes = len(self.sense_vocab)
        self.sentences, self.labels = self._load_data()

    def _load_sense_vocab(self):
        sense_vocab_file = os.path.join(self.data_path, 'sense_vocab.json')
        with open(sense_vocab_file, 'r', encoding='utf8') as f:
            sense_vocab = json.load(f)
        return sense_vocab

    def _load_data(self):
        sentences = []
        labels = []
        for file_name in os.listdir(self.data_path):
            if file_name.endswith('.xml'):
                sent = []
                lbl = []
                tree = ET.parse(os.path.join(self.data_path, file_name))
                root = tree.getroot()
                for child in root:
                    token = child.attrib['text']
                    break_level = child.attrib['break_level']
                    if 'sense' in child.attrib:
                        sense_tag = self.sense_vocab[child.attrib['sense']]
                    else:
                        # sense_tag = self.sense_vocab[token.lower()]
                        sense_tag = self.sense_vocab['unambiguous']
                    if break_level == 'PARAGRAPH_BREAK':
                        sentences.append(sent)
                        labels.append(lbl)
                        sent = [token]
                        lbl = [sense_tag]
                    else:
                        sent.append(token)
                        lbl.append(sense_tag)
                sentences.append(sent)
                labels.append(lbl)
        return sentences, labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]
