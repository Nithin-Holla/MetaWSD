import json
import os

import xml.etree.ElementTree as ET
from collections import defaultdict


def generate_sense_inventory(data_path):
    sense_inventory = defaultdict(list)
    for file_name in os.listdir(data_path):
        if file_name.endswith('.xml'):
            tree = ET.parse(os.path.join(data_path, file_name))
            root = tree.getroot()
            for child in root:
                token = child.attrib['text'].lower()
                if 'sense' in child.attrib and child.attrib['sense'] not in sense_inventory[token]:
                    sense_inventory[token].append(child.attrib['sense'])
                elif token not in sense_inventory[token]:
                    sense_inventory[token].append(token)
    return sense_inventory


def save_as_json(dict_data, json_file):
    with open(json_file, 'w', encoding='utf8') as f:
        json.dump(dict_data, f, indent=4, sort_keys=True)


def build_sense_vocab(sense_inventory_dict):
    word_to_ix = dict()
    count = 0
    for word in sense_inventory_dict.keys():
        for sense in sense_inventory_dict[word]:
            if sense not in word_to_ix:
                word_to_ix[sense] = count
                count += 1
    return word_to_ix


if __name__ == '__main__':
    data_path = '../../data/word_sense_disambigation_corpora/semcor'
    sense_inventory_file = os.path.join(data_path, 'semcor_sense_inventory.json')
    sense_vocab_file = os.path.join(data_path, 'sense_vocab.json')

    sense_inventory_dict = generate_sense_inventory(data_path)
    save_as_json(sense_inventory_dict, sense_inventory_file)

    sense_vocab = build_sense_vocab(sense_inventory_dict)
    save_as_json(sense_vocab, sense_vocab_file)
