import logging
import os
import random

import coloredlogs
import torch

from datetime import datetime

from datasets import utils
from datasets.metaphor_dataset import MetaphorDataset
from datasets.pos_dataset import POSDataset
from datasets.wsd_dataset import WSDDataset
from models.maml import MAML

CONFIG = {
    'stamp': str(datetime.now()).replace(':', '-').replace(' ', '_'),
    'meta_model': 'seq_meta_model',
    'learner_params': {
        'vocab_size': 200002,
        'hidden_size': 128,
        'num_outputs': {
            'pos': 18,
            'wsd': 45596,
            'metaphor': 2
        },
        'embed_dim': 1024,
    },
    'trained_learner': None,
    'learner_lr': 1e-2,
    'num_shots': {
        'pos': 64,
        'wsd': 64,
        'metaphor': 64
    },
    'num_updates': 1,
    'num_test_samples': {
        'pos': 64,
        'wsd': 64,
        'metaphor': 64
    },
    'num_episodes': {
        'pos': 1,
        'wsd': 1,
        'metaphor': 1
    },
    'num_meta_epochs': 5,
    'early_stopping': 10,
    'base_path': os.path.dirname(os.path.abspath(__file__)),
}

logger = logging.getLogger('MetaLearningLog')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')

if __name__ == '__main__':
    # Set seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Episodes for meta-training and meta-testing
    train_episodes, test_episodes = [], []

    # Paths for POS tagging dataset
    pos_base_path = os.path.join(CONFIG['base_path'], '../data/UD_English-EWT/')
    pos_train_path = os.path.join(pos_base_path, 'en_ewt-ud-train.conllu')
    pos_test_path = os.path.join(pos_base_path, 'en_ewt-ud-test.conllu')

    # Create POS tagging train and test dataset
    logger.info('Loading dataset for POS tagging')
    pos_train_dataset = POSDataset(pos_train_path)
    pos_test_dataset = POSDataset(pos_test_path)
    logger.info('Finished loading the dataset for POS tagging')

    # Generate episodes for POS tagging
    logger.info('Generating episodes for POS tagging')
    pos_episodes = utils.generate_episodes_from_split_datasets(train_dataset=pos_train_dataset,
                                                               test_dataset=pos_test_dataset,
                                                               n_episodes=CONFIG['num_episodes']['pos'],
                                                               n_support_examples=CONFIG['num_shots']['pos'],
                                                               n_query_examples=CONFIG['num_test_samples']['pos'],
                                                               task='pos')
    train_episodes.extend(pos_episodes)
    logger.info('Finished generating episodes for POS tagging')

    # Path for WSD dataset
    wsd_base_path = os.path.join(CONFIG['base_path'], '../data/word_sense_disambigation_corpora/semcor/')

    # Load WSD dataset
    logger.info('Loading the dataset for WSD')
    wsd_dataset = WSDDataset(wsd_base_path)
    logger.info('Finished loading the dataset for WSD')

    # Generate episodes for WSD
    logger.info('Generating episodes for WSD')
    wsd_episodes = utils.generate_episodes_from_single_dataset(dataset=wsd_dataset,
                                                               n_episodes=CONFIG['num_episodes']['wsd'],
                                                               n_support_examples=CONFIG['num_shots']['wsd'],
                                                               n_query_examples=CONFIG['num_test_samples']['wsd'],
                                                               task='wsd')
    train_episodes.extend(wsd_episodes)
    logger.info('Finished generating episodes for WSD')

    # Paths for metaphor dataset
    metaphor_base_path = os.path.join(CONFIG['base_path'], '../data/vuamc/')
    metaphor_train_path = os.path.join(metaphor_base_path, 'vuamc_corpus_train.csv')
    metaphor_test_path = os.path.join(metaphor_base_path, 'vuamc_corpus_test.csv')

    # Load metaphor train and test dataset
    logger.info('Loading the dataset for metaphor')
    metaphor_train_dataset = MetaphorDataset(metaphor_train_path)
    metaphor_test_dataset = MetaphorDataset(metaphor_test_path)
    logger.info('Finished loading the dataset for metaphor')

    # Generate episodes for metaphor
    logger.info('Generating episodes for metaphor')
    metaphor_episodes = utils.generate_episodes_from_split_datasets(train_dataset=metaphor_train_dataset,
                                                                    test_dataset=metaphor_test_dataset,
                                                                    n_episodes=CONFIG['num_episodes']['metaphor'],
                                                                    n_support_examples=CONFIG['num_shots']['metaphor'],
                                                                    n_query_examples=CONFIG['num_test_samples']['metaphor'],
                                                                    task='metaphor')
    test_episodes.extend(metaphor_episodes)
    logger.info('Finished generating episodes for metaphor')

    # Initialize meta model
    meta_learner = MAML(CONFIG)

    # Meta-training
    meta_learner.training(train_episodes)
    logger.info('Meta-learning completed')

    # Meta-testing
    meta_learner.testing(test_episodes)
    logger.info('Meta-testing completed')
