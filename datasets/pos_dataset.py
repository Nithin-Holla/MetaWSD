from torch.utils import data
from conllu import parse


POS_TAGS = {
    'ADJ': 0, 'ADV': 1, 'INTJ': 2, 'NOUN': 3,
    'PROPN': 4, 'VERB': 5, 'ADP': 6, 'AUX': 7, 'CCONJ': 8,
    'DET': 9, 'NUM': 10, 'PART': 11, 'PRON': 12,
    'SCONJ': 13, 'PUNCT': 14, 'SYM': 15, 'X': 16,
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
