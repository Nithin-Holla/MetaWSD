from torch.utils import data
from conllu import parse


POS_TAGS = {
    '<PAD>': 0, 'ADJ': 1, 'ADV': 2, 'INTJ': 3, 'NOUN': 4,
    'PROPN': 5, 'VERB': 6, 'ADP': 7, 'AUX': 8, 'CCONJ': 9,
    'DET': 10, 'NUM': 11, 'PART': 12, 'PRON': 13,
    'SCONJ': 14, 'PUNCT': 15, 'SYM': 16, 'X': 17,
}


class POSDataset(data.Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.sentences = self._load_data()

    def _load_data(self):
        with open(self.data_path, 'r', encoding='utf8') as f:
            conllu_data = f.read()
            sentence_list = parse(conllu_data)
        return sentence_list

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = [x['form'] for x in self.sentences[index]]
        labels = [POS_TAGS[x['upostag']] for x in self.sentences[index]]
        return sentence, labels
