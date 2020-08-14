# MetaWSD

This is the official code for the paper [Learning to Learn to Disambiguate: Meta-Learning for Few-Shot Word Sense Disambiguation](https://arxiv.org/abs/2004.14355).


## Getting started

- Clone the repository: `git clone git@github.com:Nithin-Holla/MetaWSD.git`.
- Create a virtual environment.
- Install the required packages: `pip install -r MetaWSD/requirements.txt`.
- Create a directory for storing the data: `mkdir data`.
- Navigate to the data directory: `cd data`.
- Clone the repository containing the data: `git clone git@github.com:google-research-datasets/word_sense_disambigation_corpora.git`.
- Navigate back: `cd ..`

## Preparing the data

- The first step is to generate the sense inventory: `python MetaWSD/scripts/wsd_gen_sense_inventory.py`.
- Next, generate episodes from the data: `python MetaWSD/scripts/generate_wsd_data.py --n_support_examples N_SUPPORT_EXAMPLES --n_query_examples N_QUERY_EXAMPLES --n_train_episodes N_TRAIN_EPISODES`.


## Training the models

The YAML configuration files for all the models are in `config/wsd`. To train a model, run `python MetaWSD/train_wsd.py --config CONFIG_FILE`. 
Training on multiple GPUs is supported for the MAML variants only. In order to use multiple GPUs, specify the flag `--multi_gpu`.
