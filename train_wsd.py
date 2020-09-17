import logging
import os
import random
import warnings
from argparse import ArgumentParser

import coloredlogs
import torch
import yaml

from datetime import datetime
from datasets import utils
from models.baseline import Baseline
from models.majority_classifier import MajorityClassifier
from models.maml import MAML
from models.nearest_neighbor import NearestNeighborClassifier
from models.proto_network import PrototypicalNetwork


logger = logging.getLogger('MetaLearningLog')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore", category=UserWarning)


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
    parser.add_argument('--multi_gpu', action='store_true')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_file)
    config['multi_gpu'] = args.multi_gpu
    logger.info('Using configuration: {}'.format(config))

    # Set seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Episodes for meta-training, meta-validation and meta-testing
    train_episodes, val_episodes, test_episodes = [], [], []

    # Directory for saving models
    os.makedirs(os.path.join(config['base_path'], 'saved_models'), exist_ok=True)

    # Path for WSD dataset
    wsd_base_path = os.path.join(config['base_path'], '../data/semcor_meta/')
    wsd_train_path = os.path.join(wsd_base_path, 'meta_train_' + str(config['num_shots']['wsd']) + '-' +
                                  str(config['num_test_samples']['wsd']))
    wsd_val_path = os.path.join(wsd_base_path, 'meta_val_' + str(config['num_shots']['wsd']) + '-' +
                                str(config['num_test_samples']['wsd']))
    wsd_test_path = os.path.join(wsd_base_path, 'meta_test_' + str(config['num_shots']['wsd']) + '-' +
                                 str(config['num_test_samples']['wsd']))

    # Generate episodes for WSD
    logger.info('Generating episodes for WSD')
    wsd_train_episodes = utils.generate_wsd_episodes(dir=wsd_train_path,
                                                     n_episodes=config['num_train_episodes']['wsd'],
                                                     n_support_examples=config['num_shots']['wsd'],
                                                     n_query_examples=config['num_test_samples']['wsd'],
                                                     task='wsd',
                                                     meta_train=True)
    wsd_val_episodes = utils.generate_wsd_episodes(dir=wsd_val_path,
                                                   n_episodes=config['num_val_episodes']['wsd'],
                                                   n_support_examples=config['num_shots']['wsd'],
                                                   n_query_examples=config['num_test_samples']['wsd'],
                                                   task='wsd',
                                                   meta_train=False)
    wsd_test_episodes = utils.generate_wsd_episodes(dir=wsd_test_path,
                                                    n_episodes=config['num_test_episodes']['wsd'],
                                                    n_support_examples=config['num_shots']['wsd'],
                                                    n_query_examples=config['num_test_samples']['wsd'],
                                                    task='wsd',
                                                    meta_train=False)
    train_episodes.extend(wsd_train_episodes)
    val_episodes.extend(wsd_val_episodes)
    test_episodes.extend(wsd_test_episodes)
    logger.info('Finished generating episodes for WSD')

    # Initialize meta learner
    if config['meta_learner'] == 'maml':
        meta_learner = MAML(config)
    elif config['meta_learner'] == 'proto_net':
        meta_learner = PrototypicalNetwork(config)
    elif config['meta_learner'] == 'baseline':
        meta_learner = Baseline(config)
    elif config['meta_learner'] == 'majority':
        meta_learner = MajorityClassifier()
    elif config['meta_learner'] == 'nearest_neighbor':
        meta_learner = NearestNeighborClassifier(config)
    else:
        raise NotImplementedError

    # Meta-training
    meta_learner.training(train_episodes, val_episodes)
    logger.info('Meta-learning completed')

    # Meta-testing
    meta_learner.testing(test_episodes)
    logger.info('Meta-testing completed')
