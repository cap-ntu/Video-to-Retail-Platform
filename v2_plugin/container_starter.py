import argparse
from pathlib import Path

import docker as docker
import yaml
from docker.types import Mount

from hysia.utils.misc import dict_to_object, object_to_dict, obtain_device

DEFAULT_GRPC_PORT = 8000
DEFAULT_HTTP_PORT = 8001


def load_service_config(service_yml_path: str):
    base_root = Path(service_yml_path).absolute().parent
    with open(service_yml_path, 'r') as f:
        service_configs = yaml.safe_load(f)

        service_config_list = list()
        for service_config in service_configs:
            service_config_list.append(dict_to_object(service_config))

    return service_config_list, base_root


def generate_config(service_config_list, base_root: Path):
    """Generate `.config.yml` for each defined service from service configuration file."""
    for config in service_config_list:
        env_config = getattr(config, 'env', object())

        config_dict = {
            'name': config.name,
            'grpc': {
                'max_workers': 8,
                'port': DEFAULT_GRPC_PORT,
            },
            'http': {
                'host': '0.0.0.0',
                'port': DEFAULT_HTTP_PORT,
            },
            'engine': object_to_dict(getattr(config, 'engine_config', None)),
            'env': {
                'conda': getattr(env_config, 'conda', None),
                'pip': getattr(env_config, 'pip', None),
                'pre_install': getattr(env_config, 'pre_install', None),
            }
        }

        service_base_root = Path(config.base_root)
        if not service_base_root.is_absolute():
            service_base_root = base_root / service_base_root

        # dict to yml
        with open(service_base_root / '.config.yml', 'w') as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)


def start_container(service_config_list, base_root: Path):
    docker_tag = 'latest-gpu'

    docker_client = docker.from_env()

    containers = list()

    for config in service_config_list:
        service_base_root = Path(config.base_root)
        if not service_base_root.is_absolute():
            service_base_root = base_root / service_base_root

        common_kwargs = {'detach': True, 'auto_remove': False, 'name': f'auto.{config.name}'}

        # set mount
        mount_common_kwargs = {'type': 'bind'}
        mounts = [
            Mount(target=f'/content/app', source=str(service_base_root), **mount_common_kwargs),
            Mount(target=f'/content/config.yml', source=str(service_base_root / '.config.yml'), **mount_common_kwargs),
            Mount(
                target=f'/content/app/predictor.py',
                source=str(service_base_root / config.predictor),
                **mount_common_kwargs
            ),
            Mount(
                target=f'/content/app/engine.py',
                source=str(service_base_root / config.engine),
                **mount_common_kwargs
            ),
        ]
        common_kwargs['mounts'] = mounts

        # port binding
        grpc_port = getattr(config, 'grpc_port', DEFAULT_GRPC_PORT)
        http_port = getattr(config, 'http_port', DEFAULT_HTTP_PORT)
        ports = {str(DEFAULT_GRPC_PORT): grpc_port, str(DEFAULT_HTTP_PORT): http_port}

        # set environment
        cuda, device_num = obtain_device(config.device)
        environment = dict()
        if cuda:
            common_kwargs['runtime'] = 'nvidia'
            environment['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            environment['CUDA_VISIBLE_DEVICES'] = device_num

        container = docker_client.containers.run(
            f'auto-serve:{docker_tag}', environment=environment, ports=ports, **common_kwargs
        )

        print(f'Service [{config.name}] started.')

        containers.append(container)

    return containers


def deploy(service_yml_path: str):
    service_config, base_root = load_service_config(service_yml_path)
    generate_config(service_config, base_root)
    start_container(service_config, base_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy and run user defined application.')
    parser.add_argument(
        'file', type=str, nargs='?', default='service.yml', help='Path to `service.yml` configuration file.'
    )

    args = parser.parse_args()
    deploy(args.file)
