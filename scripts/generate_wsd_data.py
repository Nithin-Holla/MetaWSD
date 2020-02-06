import json
import random
from collections import Counter, defaultdict
import itertools
import os
import numpy as np

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import utils
from datasets.wsd_dataset import SemCorWSDDataset


def shuffle_list(*ls):
    l = list(zip(*ls))
    random.shuffle(l)
    return zip(*l)


def generate_label_statistics(wsd_episodes, file_name):
    label_freq = {}
    for e in wsd_episodes:
        word = e.task_id.split('-')[1]
        label_freq[word] = {}
        support_labels, query_labels = [], []
        for _, _, sup_lbl in e.support_loader:
            support_labels.extend([l for l in itertools.chain(*sup_lbl) if l != -1])
        label_freq[word]['support'] = Counter(support_labels)

        for _, _, query_lbl in e.query_loader:
            query_labels.extend([l for l in itertools.chain(*query_lbl) if l != -1])
        label_freq[word]['query'] = Counter(query_labels)

    with open(file_name, 'w', encoding='utf8') as f:
        json.dump(label_freq, f, indent=4, sort_keys=True)


def update_counter_and_tracker(counter, tracker, lst):
    for key in tracker:
        tracker[key] = [x for x in tracker[key] if x not in lst]
        counter[key] = len(tracker[key])
    return counter, tracker


def fill_once(sentences, labels):
    label_counter = Counter()
    label_tracker = defaultdict(list)
    support_ids, query_ids = [], []
    for id, lbl in enumerate(labels):
        lbl = set([l for l in lbl if l != -1])
        label_counter.update(lbl)
        for l in lbl:
            label_tracker[l].append(id)
    label_traverse_order = [key for key in sorted(label_counter, key=lambda k: label_counter[k])]
    for lbl in label_traverse_order:
        if label_counter[lbl] == 1:
            picked_ids = label_tracker[lbl][0:1]
            support_ids.append(label_tracker[lbl].pop(0))
        else:
            picked_ids = label_tracker[lbl][0:2]
            support_ids.append(label_tracker[lbl].pop(0))
            query_ids.append(label_tracker[lbl].pop(0))
        # Remove the popped ids from other label trackers
        label_counter, label_tracker = update_counter_and_tracker(label_counter, label_tracker, picked_ids)

    support_entries = [{'sentence': sentences[i], 'label': labels[i]} for i in support_ids]
    query_entries = [{'sentence': sentences[i], 'label': labels[i]} for i in query_ids]
    remaining_sentences = [sentences[i] for i in range(len(sentences)) if i not in (support_ids + query_ids)]
    remaining_labels = [labels[i] for i in range(len(labels)) if i not in (support_ids + query_ids)]
    return support_entries, query_entries, remaining_sentences, remaining_labels


def split_examples(sentences, labels, n_support_examples, n_query_examples):
    sentences, labels = shuffle_list(sentences, labels)
    final_support_list, final_query_list = [], []
    for _ in range(min(n_support_examples, n_query_examples)):
        support_entries, query_entries, remaining_sentences, remaining_labels = fill_once(sentences, labels)
        final_support_list.extend(support_entries)
        final_query_list.extend(query_entries)
        sentences = remaining_sentences
        labels = remaining_labels
    return final_support_list, final_query_list


def filter_seen_sentences(examples, seen_sentences):
    filtered_examples = []
    for ex in examples:
        if ex['sentence'] not in seen_sentences:
            filtered_examples.append(ex)
    return filtered_examples


# def reorder_examples(sentences, labels):
#     sentences, labels = shuffle_list(sentences, labels)
#     # Add one sample for each of the unique labels first
#     unique_sample_ids = []
#     unique_labels = np.unique([l for l in list(itertools.chain(*labels)) if l != -1])
#     label_tracker = {k: False for k in unique_labels}
#     for id, lbl in enumerate(labels):
#         unseen_labels = set([k for k in label_tracker if label_tracker[k] is False])
#         found_labels = set([l for l in lbl if l != -1])
#         if len(unseen_labels) == 0:
#             break
#         if set.intersection(found_labels, unseen_labels):
#             label_tracker.update({k: True for k in found_labels})
#             unique_sample_ids.append(id)
#
#     sampled_sentences = [sent for (id, sent) in enumerate(sentences) if id in unique_sample_ids]
#     sampled_labels = [lbl for (id, lbl) in enumerate(labels) if id in unique_sample_ids]
#
#     # Add remaining sentences/labels
#     for id, (sent, lbl) in enumerate(zip(sentences, labels)):
#         if id not in unique_sample_ids:
#             sampled_sentences.append(sent)
#             sampled_labels.append(lbl)
#     return sampled_sentences, sampled_labels


def create_data(semcor_wsd_dataset, n_support_examples, n_query_examples, n_train_words, n_test_words, train_path,
                test_path):
    # Generate word splits
    word_splits = {k: v for (k, v) in semcor_wsd_dataset.word_splits.items() if len(v['sentences']) >
                   (n_support_examples + n_query_examples)}
    all_words = list(word_splits.keys())
    random.shuffle(all_words)

    do_test = False
    train_sentences = []
    n_good_test_episodes = 0
    for word_id, word in enumerate(all_words):
        if word_id == n_train_words:
            do_test = True
        if word_id == n_train_words + n_test_words:
            break

        word_data = []
        word_data_counter = 0
        if do_test:
            file_name = os.path.join(test_path, word + '.json')
        else:
            file_name = os.path.join(train_path, word + '.json')

        # sentences, labels = reorder_examples(word_splits[word]['sentences'], word_splits[word]['labels'])
        support_examples, query_examples = split_examples(word_splits[word]['sentences'], word_splits[word]['labels'],
                                                          n_support_examples, n_query_examples)
        if not do_test:
            train_sentences.extend([ex['sentence'] for ex in support_examples])

        if do_test:
            query_examples = filter_seen_sentences(query_examples, train_sentences)

        word_data = support_examples + query_examples
        utils.write_json(word_data, file_name)

    #     for sent, label in zip(sentences, labels):
    #         if do_test and word_data_counter > n_support_examples and sent in train_sentences:
    #             continue
    #         word_data.append({'sentence': sent, 'label': label})
    #         word_data_counter += 1
    #         if not do_test:
    #             train_sentences.append(sent)
    #         # if not do_test and len(word_data[word]) == n_support_examples + n_query_examples:
    #         #     break
    #
    #     if do_test and len(word_data) >= (n_support_examples + n_query_examples):
    #         n_good_test_episodes += 1
    #
    #     if do_test:
    #         print('{} -> {}'.format(word, len(word_data)))
    #     # random.shuffle(word_data)
    #     utils.write_json(word_data, file_name)
    #
    # print('Good test episodes', n_good_test_episodes)


if __name__ == '__main__':
    random.seed(42)

    n_episodes = 600
    n_support_examples = 16
    n_query_examples = 16
    n_train_words = 400
    n_test_words = 200

    # Path for WSD dataset
    base_path = os.path.dirname(os.path.abspath(__file__))
    semcor_wsd_base_path = os.path.join(base_path, '../../data/word_sense_disambigation_corpora/semcor/')

    # Path for writing the new data
    write_path = os.path.join(base_path, '../../data/semcor_meta')
    os.makedirs(write_path, exist_ok=True)
    train_path = os.path.join(write_path, 'meta_train_' + str(n_support_examples) + '-' + str(n_query_examples))
    test_path = os.path.join(write_path, 'meta_test_' + str(n_support_examples) + '-' + str(n_query_examples))
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Load WSD dataset
    semcor_wsd_dataset = SemCorWSDDataset(semcor_wsd_base_path)

    # Create data grouped by word and write to disk
    create_data(semcor_wsd_dataset, n_support_examples, n_query_examples, n_train_words, n_test_words, train_path,
                test_path)

    # Label statistics
    wsd_train_episodes = utils.generate_wsd_episodes(dir=train_path,
                                                     n_episodes=n_train_words,
                                                     n_support_examples=n_support_examples,
                                                     n_query_examples=n_query_examples,
                                                     task='wsd',
                                                     meta_train=True)
    wsd_test_episodes = utils.generate_wsd_episodes(dir=test_path,
                                                    n_episodes=n_test_words,
                                                    n_support_examples=n_support_examples,
                                                    n_query_examples=n_query_examples,
                                                    task='wsd',
                                                    meta_train=False)
    generate_label_statistics(wsd_train_episodes, 'meta_train_stat.json')
    generate_label_statistics(wsd_test_episodes, 'meta_test_stat.json')
