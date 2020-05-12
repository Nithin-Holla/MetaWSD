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
from datasets.metaphor_dataset import MetaphorDataset, MetaphorClassificationDataset
from models.baseline import Baseline
from models.majority_classifier import MajorityClassifier
from models.maml import MAML
from models.nearest_neighbor import NearestNeighborClassifier
from models.proto_network import PrototypicalNetwork


logger = logging.getLogger('MetaLearningLog')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')

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
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_file)
    logger.info('Using configuration: {}'.format(config))

    # Set seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Episodes for meta-training, meta-validation and meta-testing
    train_episodes, val_episodes, test_episodes = [], [], []

    # Directory for saving models
    os.makedirs(os.path.join(config['base_path'], 'saved_models'), exist_ok=True)

    # Paths for metaphor dataset
    metaphor_base_path = os.path.join(config['base_path'], '../data/vuamc/')
    metaphor_train_path = os.path.join(metaphor_base_path, 'VUA_cls_train.csv')
    metaphor_test_path = os.path.join(metaphor_base_path, 'VUA_cls_test.csv')

    # Load metaphor train and test dataset
    logger.info('Loading the dataset for metaphor')
    metaphor_train_dataset = MetaphorClassificationDataset(metaphor_train_path, config['learner_params']['num_outputs']['metaphor'])
    metaphor_test_dataset = MetaphorClassificationDataset(metaphor_test_path, config['learner_params']['num_outputs']['metaphor'])
    logger.info('Finished loading the dataset for metaphor')

    # Generate episodes for metaphor
    logger.info('Generating episodes for metaphor')
    metaphor_episodes = utils.generate_metaphor_cls_episodes(train_dataset=metaphor_train_dataset,
                                                             test_dataset=metaphor_test_dataset,
                                                             n_support_examples=config['num_shots']['metaphor'],
                                                             task='metaphor')
    test_episodes.extend(metaphor_episodes)
    logger.info('Finished generating {} episodes for metaphor'.format(len(metaphor_episodes)))

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

    # Meta-testing
    meta_learner.testing(test_episodes)
    logger.info('Meta-testing completed')
