from pathlib import Path

import yaml

from utils.misc import dict_to_object

WEIGHT_DIR = Path(__file__).absolute().parents[2] / 'weights'

CONFIG_DIR = Path(__file__).absolute().parents[1] / 'config'

# load config
with open(CONFIG_DIR / 'weight_config.yml', 'r') as stream:
    try:
        config = yaml.safe_load(stream)
        config = dict_to_object(config)
    except yaml.YAMLError as e:
        print(e)
        exit(1)

# load device placement
with open(CONFIG_DIR / 'device_placement.yml') as stream:
    try:
        device_config = yaml.safe_load(stream)
        device_config = dict_to_object(device_config)
    except yaml.YAMLError as e:
        print(e)
        exit(1)

__all__ = ['WEIGHT_DIR', 'config', 'device_config']
