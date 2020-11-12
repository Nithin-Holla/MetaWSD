# MetaWSD

This is the official code for the paper [Learning to Learn to Disambiguate: Meta-Learning for Few-Shot Word Sense Disambiguation](https://arxiv.org/abs/2004.14355), published at Findings of EMNLP.


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

- The YAML configuration files for all the models are in `config/wsd`. To train a model, run `python MetaWSD/train_wsd.py --config CONFIG_FILE`.
- For using the non-episodic baseline, switch to the `baseline` branch. For all the other models, use the code in the `master` branch.
- Training on multiple GPUs is supported for the MAML variants only. In order to use multiple GPUs, specify the flag `--multi_gpu`.


## Troubleshooting

If you have a `RuntimeError` with Proto(FO)MAML and BERT, you can install the `higher` library from this fork: [https://github.com/Nithin-Holla/higher](https://github.com/Nithin-Holla/higher), which has a temporary fix for this. Also, replace `diffopt.step(loss)` with `diffopt.step(loss, retain_graph=True)` in `models/seq_meta.py`.


## Citation

If you use this code repository, please consider citing the paper:
```bib
@inproceedings{holla-etal-2020-learning,
    title = "Learning to Learn to Disambiguate: Meta-Learning for Few-Shot Word Sense Disambiguation",
    author = "Holla, Nithin and Mishra, Pushkar and Yannakoudakis, Helen and Shutova, Ekaterina",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.405",
    pages = "4517--4533"
}
```