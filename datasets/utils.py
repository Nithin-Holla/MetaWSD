import glob
import json
import os
import random
import torch

import numpy as np

from torch.utils import data
from torch.utils.data import Subset
from transformers import AlbertTokenizer

from datasets.episode import Episode
from datasets.wsd_dataset import WordWSDDataset, MetaWSDDataset
from datasets.fewrel_dataset import FewRelDataset, FewRelSubset


def write_json(json_dict, file_name):
    with open(file_name, 'w', encoding='utf8') as f:
        json.dump(json_dict, f, indent=4)


def read_json(file_name):
    with open(file_name, 'r', encoding='utf8') as f:
        json_dict = json.load(f)
    return json_dict


def get_max_batch_len(batch):
    return max([len(x[0]) for x in batch])


def prepare_batch(batch):
    max_len = get_max_batch_len(batch)
    x = []
    lengths = []
    y = []
    for inp_seq, target_seq in batch:
        lengths.append(len(inp_seq))
        target_seq = target_seq + [-1] * (max_len - len(target_seq))
        x.append(inp_seq)
        y.append(target_seq)
    return x, lengths, y


def collate_fewrel(data):
    batch_data = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_label = []
    data_points, labels = zip(*data)
    for i in range(len(data_points)):
        for k in data_points[i]:
            batch_data[k].append(data_points[i][k])
        batch_label.append(labels[i])
    for k in batch_data:
        batch_data[k] = torch.stack(batch_data[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_data, batch_label


def prepare_task_batch(batch):
    return batch


def generate_episodes_from_split_datasets(train_dataset, test_dataset, n_episodes, n_support_examples, n_query_examples, task):
    if n_episodes * n_support_examples > train_dataset.__len__() or n_episodes * n_query_examples > test_dataset.__len__():
        raise Exception('Not enough data available')

    train_indices = list(range(train_dataset.__len__()))
    random.shuffle(train_indices)

    test_indices = list(range(test_dataset.__len__()))
    random.shuffle(test_indices)

    train_start_index, test_start_index = 0, 0
    episodes = []
    for e in range(n_episodes):
        train_subset = data.Subset(train_dataset, train_indices[train_start_index: train_start_index + n_support_examples])
        support_loader = data.DataLoader(train_subset, batch_size=n_support_examples, collate_fn=prepare_batch)
        train_start_index += n_support_examples
        test_subset = data.Subset(test_dataset, test_indices[test_start_index: test_start_index + n_query_examples])
        query_loader = data.DataLoader(test_subset, batch_size=n_query_examples, collate_fn=prepare_batch)
        test_start_index += n_query_examples
        episode = Episode(support_loader, query_loader, task, task, train_dataset.n_classes)
        episodes.append(episode)
    return episodes


def generate_episodes_from_single_dataset(dataset, n_episodes, n_support_examples, n_query_examples, task):
    if n_episodes * (n_support_examples + n_query_examples) > dataset.__len__():
        raise Exception('Not enough data available')

    indices = list(range(dataset.__len__()))
    random.shuffle(indices)

    start_index = 0
    episodes = []
    for e in range(n_episodes):
        train_subset = data.Subset(dataset, indices[start_index: start_index + n_support_examples])
        support_loader = data.DataLoader(train_subset, batch_size=n_support_examples, collate_fn=prepare_batch)
        start_index += n_support_examples
        test_subset = data.Subset(dataset, indices[start_index: start_index + n_query_examples])
        query_loader = data.DataLoader(test_subset, batch_size=n_query_examples, collate_fn=prepare_batch)
        start_index += n_query_examples
        episode = Episode(support_loader, query_loader, task, task, dataset.n_classes)
        episodes.append(episode)
    return episodes


def generate_full_query_episode(train_dataset, test_dataset, n_support_examples, task, batch_size=32):
    train_indices = list(range(train_dataset.__len__()))
    random.shuffle(train_indices)
    train_subset = data.Subset(train_dataset, train_indices[0:n_support_examples])
    support_loader = data.DataLoader(train_subset, batch_size=n_support_examples, collate_fn=prepare_batch)
    query_loader = data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=prepare_batch)
    episode = Episode(support_loader, query_loader, task, task, train_dataset.n_classes)
    return [episode]


def generate_semcor_wsd_episodes(wsd_dataset, n_episodes, n_support_examples, n_query_examples, task):
    word_splits = {k: v for (k, v) in wsd_dataset.word_splits.items() if len(v['sentences']) >
                   (n_support_examples + n_query_examples)}

    if n_episodes > len(word_splits):
        raise Exception('Not enough data available to generate {} episodes'.format(n_episodes))

    episodes = []
    for word in word_splits.keys():
        if len(episodes) == n_episodes:
            break
        indices = list(range(len(word_splits[word]['sentences'])))
        random.shuffle(indices)
        start_index = 0
        train_subset = WordWSDDataset(sentences=[word_splits[word]['sentences'][i] for i in indices[start_index: start_index + n_support_examples]],
                                      labels=[word_splits[word]['labels'][i] for i in indices[start_index: start_index + n_support_examples]],
                                      n_classes=len(wsd_dataset.sense_inventory[word]))
        support_loader = data.DataLoader(train_subset, batch_size=n_support_examples, collate_fn=prepare_batch)
        start_index += n_support_examples
        test_subset = WordWSDDataset(sentences=[word_splits[word]['sentences'][i] for i in indices[start_index: start_index + n_query_examples]],
                                     labels=[word_splits[word]['labels'][i] for i in indices[start_index: start_index + n_query_examples]],
                                     n_classes=len(wsd_dataset.sense_inventory[word]))
        query_loader = data.DataLoader(test_subset, batch_size=n_query_examples, collate_fn=prepare_batch)
        episode = Episode(support_loader=support_loader,
                          query_loader=query_loader,
                          base_task=task,
                          task_id=task + '-' + word,
                          n_classes=train_subset.n_classes)
        episodes.append(episode)
    return episodes


def generate_wsd_episodes(dir, n_episodes, n_support_examples, n_query_examples, task, meta_train=True):
    episodes = []
    for file_name in glob.glob(os.path.join(dir, '*.json')):
        if len(episodes) == n_episodes:
            break
        word = file_name.split(os.sep)[-1].split('.')[0]
        word_wsd_dataset = MetaWSDDataset(file_name)
        train_subset = Subset(word_wsd_dataset, range(0, n_support_examples))
        support_loader = data.DataLoader(train_subset, batch_size=n_support_examples, collate_fn=prepare_batch)
        if meta_train:
            test_subset = Subset(word_wsd_dataset, range(n_support_examples, n_support_examples + n_query_examples))
        else:
            test_subset = Subset(word_wsd_dataset, range(n_support_examples, len(word_wsd_dataset)))
        query_loader = data.DataLoader(test_subset, batch_size=n_query_examples, collate_fn=prepare_batch)
        episode = Episode(support_loader=support_loader,
                          query_loader=query_loader,
                          base_task=task,
                          task_id=task + '-' + word,
                          n_classes=word_wsd_dataset.n_classes)
        episodes.append(episode)
    return episodes


def fewrel_tokenize(raw_tokens, pos_head, pos_tail, max_length=128):
    # token -> index
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    tokens = ['[CLS]']
    cur_pos = 0
    pos1_in_index = 1
    pos2_in_index = 1
    for token in raw_tokens:
        token = token.lower()
        if cur_pos == pos_head[0]:
            tokens.append('[unused0]')
            pos1_in_index = len(tokens)
        if cur_pos == pos_tail[0]:
            tokens.append('[unused1]')
            pos2_in_index = len(tokens)
        tokens += tokenizer.tokenize(token)
        if cur_pos == pos_head[-1]:
            tokens.append('[unused2]')
        if cur_pos == pos_tail[-1]:
            tokens.append('[unused3]')
        cur_pos += 1
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)

    # padding
    while len(indexed_tokens) < max_length:
        indexed_tokens.append(0)
    indexed_tokens = indexed_tokens[:max_length]

    # mask
    mask = np.zeros((max_length), dtype=np.int32)
    mask[:len(tokens)] = 1

    pos1_in_index = min(max_length, pos1_in_index)
    pos2_in_index = min(max_length, pos2_in_index)

    return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask


def generate_fewrel_episodes(dir, name, N, K, Q, n_episodes, task):
    episodes = []
    fewrel_dataset = FewRelDataset(name, fewrel_tokenize, N, K, Q, na_rate=0, root=dir)
    data_loader = data.DataLoader(dataset=fewrel_dataset, batch_size=1, shuffle=False, collate_fn=prepare_task_batch)
    data_loader = iter(data_loader)
    for i in range(n_episodes):
        support_set, query_set, labels = next(data_loader)[0]
        support_subset = FewRelSubset(support_set, labels)
        support_loader = data.DataLoader(support_subset, batch_size=N*K, collate_fn=collate_fewrel)
        query_subset = FewRelSubset(query_set, labels)
        query_loader = data.DataLoader(query_subset, batch_size=N*Q, collate_fn=collate_fewrel)
        episode = Episode(support_loader=support_loader,
                          query_loader=query_loader,
                          base_task=task,
                          task_id=task + '-' + str(i),
                          n_classes=N)
        episodes.append(episode)
    return episodes
