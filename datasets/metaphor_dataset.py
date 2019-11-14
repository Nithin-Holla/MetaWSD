import csv

from torch.utils import data


class MetaphorDataset(data.Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.sentences, self.labels = self._load_data()

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
