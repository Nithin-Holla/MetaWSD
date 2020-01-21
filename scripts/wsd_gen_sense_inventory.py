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
                if 'sense' in child.attrib:
                    lemma = child.attrib['lemma'].lower()
                    if child.attrib['sense'] not in sense_inventory[lemma]:
                        sense_inventory[lemma].append(child.attrib['sense'])
    sense_inventory = {k: v for (k, v) in sense_inventory.items() if len(v) > 1}
    return sense_inventory


def save_as_json(dict_data, json_file):
    with open(json_file, 'w', encoding='utf8') as f:
        json.dump(dict_data, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, '../../data/word_sense_disambigation_corpora/semcor')
    sense_inventory_file = os.path.join(data_path, 'semcor_sense_inventory.json')
    sense_vocab_file = os.path.join(data_path, 'sense_vocab.json')

    sense_inventory_dict = generate_sense_inventory(data_path)
    save_as_json(sense_inventory_dict, sense_inventory_file)
