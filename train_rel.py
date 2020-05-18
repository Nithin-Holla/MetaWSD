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
from models.maml import MAML
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

    # Episodes for meta-training, meta-validation
    train_episodes, val_episodes = [], []

    # Directory for saving models
    os.makedirs(os.path.join(config['base_path'], 'saved_models'), exist_ok=True)

    # Path for WSD dataset
    fewrel_base_path = os.path.join(config['base_path'], '../data/FewRel/')

    # Generate episodes for WSD
    logger.info('Generating episodes from FewRel')
    fewrel_train_episodes = utils.generate_fewrel_episodes(dir=fewrel_base_path,
                                                           name='train_wiki',
                                                           N=config['learner_params']['num_outputs']['rel'],
                                                           K=config['num_shots']['rel'],
                                                           Q=config['num_shots']['rel'],
                                                           n_episodes=config['num_train_episodes']['rel'],
                                                           task='rel')
    fewrel_val_episodes = utils.generate_fewrel_episodes(dir=fewrel_base_path,
                                                         name='val_wiki',
                                                         N=config['learner_params']['num_outputs']['rel'],
                                                         K=config['num_shots']['rel'],
                                                         Q=config['num_shots']['rel'],
                                                         n_episodes=config['num_train_episodes']['rel'],
                                                         task='rel')
    train_episodes.extend(fewrel_train_episodes)
    val_episodes.extend(fewrel_val_episodes)
    logger.info('Finished generating episodes for FewRel')

    # Initialize meta learner
    if config['meta_learner'] == 'maml':
        meta_learner = MAML(config)
    elif config['meta_learner'] == 'proto_net':
        meta_learner = PrototypicalNetwork(config)
    else:
        raise NotImplementedError

    # Meta-training
    meta_learner.training(train_episodes, val_episodes)
    logger.info('Meta-learning completed')
