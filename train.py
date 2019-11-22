import logging
import os
import random
from argparse import ArgumentParser

import coloredlogs
import torch
import yaml

from datetime import datetime
from datasets import utils
from datasets.metaphor_dataset import MetaphorDataset
from datasets.pos_dataset import POSDataset
from datasets.wsd_dataset import WSDDataset
from models.baseline import Baseline
from models.maml import MAML
from models.proto_network import PrototypicalNetwork


logger = logging.getLogger('MetaLearningLog')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    config['base_path'] = os.path.dirname(os.path.abspath(__file__))
    config['stamp'] = str(datetime.now()).replace(':', '-').replace(' ', '_')
    return config


if __name__ == '__main__':

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument('--config', dest='config_file', type=str, help='Configuration file', required=True)
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_file)

    # Set seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Episodes for meta-training and meta-testing
    train_episodes, test_episodes = [], []

    # Directory for saving models
    os.makedirs(os.path.join(config['base_path'], 'saved_models'), exist_ok=True)

    # Paths for POS tagging dataset
    pos_base_path = os.path.join(config['base_path'], '../data/UD_English-EWT/')
    pos_train_path = os.path.join(pos_base_path, 'en_ewt-ud-train.conllu')
    pos_test_path = os.path.join(pos_base_path, 'en_ewt-ud-test.conllu')

    # Create POS tagging train and test dataset
    logger.info('Loading dataset for POS tagging')
    pos_train_dataset = POSDataset(pos_train_path)
    pos_test_dataset = POSDataset(pos_test_path)
    logger.info('Finished loading the dataset for POS tagging')

    # Generate episodes for POS tagging
    logger.info('Generating episodes for POS tagging')
    if config['num_test_samples']['pos'] == 'all':
        pos_episodes = utils.generate_full_query_episode(train_dataset=pos_train_dataset,
                                                         test_dataset=pos_test_dataset,
                                                         n_support_examples=config['num_shots']['pos'],
                                                         task='pos')
    else:
        pos_episodes = utils.generate_episodes_from_split_datasets(train_dataset=pos_train_dataset,
                                                                   test_dataset=pos_test_dataset,
                                                                   n_episodes=config['num_episodes']['pos'],
                                                                   n_support_examples=config['num_shots']['pos'],
                                                                   n_query_examples=config['num_test_samples']['pos'],
                                                                   task='pos')
    train_episodes.extend(pos_episodes)
    logger.info('Finished generating episodes for POS tagging')

    # Path for WSD dataset
    wsd_base_path = os.path.join(config['base_path'], '../data/word_sense_disambigation_corpora/semcor/')

    # Load WSD dataset
    logger.info('Loading the dataset for WSD')
    wsd_dataset = WSDDataset(wsd_base_path)
    logger.info('Finished loading the dataset for WSD')

    # Generate episodes for WSD
    logger.info('Generating episodes for WSD')
    wsd_episodes = utils.generate_episodes_from_single_dataset(dataset=wsd_dataset,
                                                               n_episodes=config['num_episodes']['wsd'],
                                                               n_support_examples=config['num_shots']['wsd'],
                                                               n_query_examples=config['num_test_samples']['wsd'],
                                                               task='wsd')
    train_episodes.extend(wsd_episodes)
    logger.info('Finished generating episodes for WSD')

    # Paths for metaphor dataset
    metaphor_base_path = os.path.join(config['base_path'], '../data/vuamc/')
    metaphor_train_path = os.path.join(metaphor_base_path, 'vuamc_corpus_train.csv')
    metaphor_test_path = os.path.join(metaphor_base_path, 'vuamc_corpus_test.csv')

    # Load metaphor train and test dataset
    logger.info('Loading the dataset for metaphor')
    metaphor_train_dataset = MetaphorDataset(metaphor_train_path)
    metaphor_test_dataset = MetaphorDataset(metaphor_test_path)
    logger.info('Finished loading the dataset for metaphor')

    # Generate episodes for metaphor
    logger.info('Generating episodes for metaphor')
    if config['num_test_samples']['metaphor'] == 'all':
        metaphor_episodes = utils.generate_full_query_episode(train_dataset=metaphor_train_dataset,
                                                              test_dataset=metaphor_test_dataset,
                                                              n_support_examples=config['num_shots']['metaphor'],
                                                              task='metaphor')
    else:
        metaphor_episodes = utils.generate_episodes_from_split_datasets(train_dataset=metaphor_train_dataset,
                                                                        test_dataset=metaphor_test_dataset,
                                                                        n_episodes=config['num_episodes']['metaphor'],
                                                                        n_support_examples=config['num_shots']['metaphor'],
                                                                        n_query_examples=config['num_test_samples']['metaphor'],
                                                                        task='metaphor')
    test_episodes.extend(metaphor_episodes)
    logger.info('Finished generating episodes for metaphor')

    # Initialize meta learner
    if config['meta_learner'] == 'maml':
        meta_learner = MAML(config)
    elif config['meta_learner'] == 'proto_net':
        meta_learner = PrototypicalNetwork(config)
    elif config['meta_learner'] == 'baseline':
        meta_learner = Baseline(config)
    else:
        raise Exception('Unsupported model type')

    # Meta-training
    meta_learner.training(train_episodes)
    logger.info('Meta-learning completed')

    # Meta-testing
    meta_learner.testing(test_episodes)
    logger.info('Meta-testing completed')
