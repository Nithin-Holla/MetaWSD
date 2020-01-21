import random

from torch.utils import data

from datasets.episode import Episode
from datasets.wsd_dataset import WordWSDDataset


def get_max_batch_len(batch):
    return max([len(x[0]) for x in batch])


def prepare_batch(batch):
    max_len = get_max_batch_len(batch)
    x = []
    lengths = []
    y = []
    for inp_seq, target_seq in batch:
        lengths.append(len(inp_seq))
        inp_seq = inp_seq + ['<PAD>'] * (max_len - len(inp_seq))
        target_seq = target_seq + [-1] * (max_len - len(target_seq))
        x.append(inp_seq)
        y.append(target_seq)
    return x, lengths, y


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


def generate_wsd_episodes(wsd_dataset, n_episodes, n_support_examples, n_query_examples, task):
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

