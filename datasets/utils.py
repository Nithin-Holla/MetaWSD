import glob
import itertools
import json
import os
import random

from torch.utils import data
from torch.utils.data import Subset

from datasets.episode import Episode
from datasets.metaphor_dataset import WordMetaphorDataset
from datasets.wsd_dataset import WordWSDDataset, MetaWSDDataset


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


def generate_metaphor_cls_episodes(train_dataset, test_dataset, n_support_examples, task, batch_size=32):
    episodes = []
    for verb in test_dataset.word_splits:
        if verb not in train_dataset.word_splits:
            continue

        non_metaphor_idx, metaphor_idx = [], []
        for idx, lbl in enumerate(train_dataset.word_splits[verb]['labels']):
            if 0 in lbl:
                non_metaphor_idx.append(idx)
            elif 1 in lbl:
                metaphor_idx.append(idx)

        if len(non_metaphor_idx) == 0 or len(metaphor_idx) == 0 or len(non_metaphor_idx) + len(metaphor_idx) < n_support_examples:
            continue

        sampled_idx = []
        random.shuffle(non_metaphor_idx)
        random.shuffle(metaphor_idx)
        while len(sampled_idx) != n_support_examples:
            if len(metaphor_idx) != 0:
                sampled_idx.append(metaphor_idx.pop())
            if len(non_metaphor_idx) != 0:
                sampled_idx.append(non_metaphor_idx.pop())

        support_sentences = [train_dataset.word_splits[verb]['sentences'][i] for i in sampled_idx]
        support_labels = [train_dataset.word_splits[verb]['labels'][i] for i in sampled_idx]
        train_subset = WordMetaphorDataset(sentences=support_sentences,
                                           labels=support_labels,
                                           n_classes=train_dataset.n_classes)
        support_loader = data.DataLoader(train_subset, batch_size=n_support_examples, collate_fn=prepare_batch)
        test_subset = WordMetaphorDataset(sentences=test_dataset.word_splits[verb]['sentences'],
                                          labels=test_dataset.word_splits[verb]['labels'],
                                          n_classes=test_dataset.n_classes)
        query_loader = data.DataLoader(test_subset, batch_size=batch_size, collate_fn=prepare_batch)
        episode = Episode(support_loader=support_loader,
                          query_loader=query_loader,
                          base_task=task,
                          task_id=task + '-' + verb,
                          n_classes=train_dataset.n_classes)
        episodes.append(episode)
    return episodes


def generate_metaphor_episode(train_dataset, test_dataset, n_support_examples, task, batch_size=32):
    metaphor_idx, non_metaphor_idx = [], []
    for idx, lbl in enumerate(train_dataset.labels):
        if 0 in lbl and len(non_metaphor_idx) < n_support_examples // 2:
            non_metaphor_idx.append(idx)
        elif 1 in lbl and len(metaphor_idx) < n_support_examples // 2:
            metaphor_idx.append(idx)
        if len(non_metaphor_idx) == n_support_examples // 2 and len(metaphor_idx) == n_support_examples // 2:
            break

    sampled_idx = metaphor_idx + non_metaphor_idx
    train_subset = data.Subset(train_dataset, sampled_idx)
    support_loader = data.DataLoader(train_subset, batch_size=n_support_examples, collate_fn=prepare_batch)
    query_loader = data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=prepare_batch)
    episode = Episode(support_loader=support_loader,
                      query_loader=query_loader,
                      base_task=task,
                      task_id=task,
                      n_classes=train_dataset.n_classes)
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
