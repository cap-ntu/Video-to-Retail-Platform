from pathlib import Path

import yaml

from utils.misc import dict_to_object

WEIGHT_DIR = Path(__file__).absolute().parents[2] / 'weights'

# load config
with open(Path(__file__).absolute().parents[1] / 'config/weight_config.yml', 'r') as stream:
    try:
        config = yaml.safe_load(stream)
        config = dict_to_object(config)
    except yaml.YAMLError as e:
        print(e)
        exit(1)

__all__ = ['WEIGHT_DIR', 'config']
