import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json


class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """

    def __init__(self, name, tokenizer, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.tokenizer = tokenizer

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.tokenizer(item['tokens'], item['h'][2][0], item['t'][2][0])
        return word, pos1, pos2, mask

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes,
                                 self.classes))

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                list(range(len(self.json_data[class_name]))),
                self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, mask = self.__getraw__(
                    self.json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, mask)
                else:
                    self.__additem__(query_set, word, pos1, pos2, mask)
                count += 1

            query_label += [i] * self.Q

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(
                list(range(len(self.json_data[cur_class]))),
                1, False)[0]
            word, pos1, pos2, mask = self.__getraw__(
                self.json_data[cur_class][index])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            self.__additem__(query_set, word, pos1, pos2, mask)
        query_label += [self.N] * Q_na

        return support_set, query_set, query_label

    def __len__(self):
        return 1000000000


class FewRelSubset(data.Dataset):
    def __init__(self, data_dict, labels):
        self.word = data_dict['word']
        self.pos1 = data_dict['pos1']
        self.pos2 = data_dict['pos2']
        self.mask = data_dict['mask']
        self.labels = labels

    def __getitem__(self, index):
        data = {'word': self.word[index], 'pos1': self.pos1[index], 'pos2': self.pos2[index], 'mask': self.mask[index]}
        label = self.labels[index]
        return data, label

    def __len__(self):
        return len(self.labels)
