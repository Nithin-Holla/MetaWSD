import logging
import os
import torch
import warnings

import pandas as pd
from argparse import ArgumentParser

import coloredlogs
import yaml
import numpy as np

from datetime import datetime

from datasets import utils
from models.baseline import Baseline
from models.majority_classifier import MajorityClassifier
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
    parser.add_argument('--learner_lr', type=float, help='Learner learning rate', default=0.1)
    parser.add_argument('--meta_lr', type=float, help='Meta learning rate', default=0.01)
    parser.add_argument('--hidden_size', type=int, help='Hidden size', default=256)
    parser.add_argument('--num_updates', type=int, help='Number of updates', default=5)
    parser.add_argument('--n_runs', type=int, help='Number of runs to average over', default=3)
    parser.add_argument('--results_file', type=str, help='File name of the results file', default='results.csv')
    parser.add_argument('--dropout_ratio', type=float, help='Dropout ratio', default=0)
    parser.add_argument('--meta_weight_decay', type=float, help='Meta weight decay', default=0)
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_file)

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
    logger.info('Finished generating episodes for WSD')

    # Update the config
    f1s = []
    config['learner_params']['hidden_size'] = args.hidden_size
    config['learner_params']['dropout_ratio'] = args.dropout_ratio
    config['learner_lr'] = args.learner_lr
    if config['meta_learner'] == 'maml' or config['meta_learner'] == 'proto_net':
        config['meta_lr'] = args.meta_lr
        config['meta_weight_decay'] = args.meta_weight_decay
    else:
        config['meta_lr'] = 0
        config['meta_weight_decay'] = 0
    if config['meta_learner'] == 'maml':
        config['num_updates'] = args.num_updates
    else:
        config['num_updates'] = 1
    logger.info('Using configuration: {}'.format(config))

    if config['meta_learner'] == 'maml':
        if config['fomaml'] and config['proto_maml']:
            model_name = 'ProtoFOMAML'
        elif config['fomaml'] and not config['proto_maml']:
            model_name = 'FOMAML'
        elif not config['fomaml'] and config['proto_maml']:
            model_name = 'ProtoMAML'
        elif not config['fomaml'] and not config['proto_maml']:
            model_name = 'MAML'
    else:
        model_name = config['meta_learner']

    run_dict = {'model_name': model_name, 'learner_lr': config['learner_lr'], 'meta_lr': config['meta_lr'],
                'hidden_size': config['learner_params']['hidden_size'], 'num_updates': config['num_updates'],
                'dropout_ratio': config['learner_params']['dropout_ratio'],
                'meta_weight_decay': config['meta_weight_decay']}
    for i in range(args.n_runs):
        torch.manual_seed(42 + i)

        # Initialize meta learner
        if config['meta_learner'] == 'maml':
            meta_learner = MAML(config)
        elif config['meta_learner'] == 'proto_net':
            meta_learner = PrototypicalNetwork(config)
        elif config['meta_learner'] == 'baseline':
            meta_learner = Baseline(config)
        elif config['meta_learner'] == 'majority':
            meta_learner = MajorityClassifier()
        else:
            raise NotImplementedError

        logger.info('Run {}'.format(i + 1))
        f1 = meta_learner.training(wsd_train_episodes, wsd_val_episodes)
        run_dict['val_' + str(i+1) + '_f1'] = f1
        f1s.append(f1)
    avg_f1 = np.mean(f1s)
    run_dict['avg_f1'] = avg_f1
    logger.info('Got average validation F1: {}'.format(avg_f1))

    results_columns = ['model_name', 'learner_lr', 'meta_lr', 'hidden_size', 'num_updates', 'dropout_ratio', 'meta_weight_decay'] \
                      + ['val_' + str(k) + '_f1' for k in range(1, args.n_runs + 1)] + ['avg_f1']
    if os.path.isfile(args.results_file):
        results_df = pd.read_csv(args.results_file)
        results_df = results_df.append(run_dict, ignore_index=True)
    else:
        results_df = pd.DataFrame(run_dict, index=[0])
    results_df.to_csv(args.results_file, index=False, columns=results_columns)
