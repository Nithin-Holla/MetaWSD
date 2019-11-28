import json
import os

import xml.etree.ElementTree as ET
from collections import defaultdict, OrderedDict


def generate_sense_inventory(data_path):
    sense_inventory = defaultdict(list)
    sense_freq = defaultdict(int)
    for file_name in os.listdir(data_path):
        if file_name.endswith('.xml'):
            tree = ET.parse(os.path.join(data_path, file_name))
            root = tree.getroot()
            for child in root:
                token = child.attrib['text'].lower()
                if 'sense' in child.attrib:
                    sense_freq[child.attrib['sense']] += 1
                    if child.attrib['sense'] not in sense_inventory[token]:
                        sense_inventory[token].append(child.attrib['sense'])
                else:
                    sense_freq[token] += 1
                    if token not in sense_inventory[token]:
                        sense_inventory[token].append(token)
    return sense_inventory, sense_freq


def save_as_json(dict_data, json_file):
    with open(json_file, 'w', encoding='utf8') as f:
        json.dump(dict_data, f, indent=4, sort_keys=True)


def build_sense_vocab(sense_freq):
    word_to_ix = OrderedDict()
    # word_to_ix['unambiguous'] = 0
    # count = 1
    count = 0
    sense_freq_ordered = OrderedDict(sorted(sense_freq.items(), key=lambda x: x[1], reverse=True))
    for sense in sense_freq_ordered.keys():
        if sense not in word_to_ix:
            word_to_ix[sense] = count
            count += 1
    return word_to_ix


if __name__ == '__main__':
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, '../../data/word_sense_disambigation_corpora/semcor')
    sense_inventory_file = os.path.join(data_path, 'semcor_sense_inventory.json')
    sense_vocab_file = os.path.join(data_path, 'sense_vocab.json')

    sense_inventory_dict, sense_freq = generate_sense_inventory(data_path)
    save_as_json(sense_inventory_dict, sense_inventory_file)

    sense_vocab = build_sense_vocab(sense_freq)
    save_as_json(sense_vocab, sense_vocab_file)
