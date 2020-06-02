import os
import shlex
import subprocess
from pathlib import Path

from utils import load_config


def fix_eol():
    args = shlex.split(r'find ./ -type f -exec sed -i -e "s/^M$//" {} \;')
    subprocess.call(args)


def install_environment(config):
    conda_file_path = config.conda
    pip_file_path = config.pip

    # install Conda environment
    if conda_file_path is not None:
        conda_file_path = Path(conda_file_path).absolute()
        if not conda_file_path.is_file():
            raise FileNotFoundError(f'Conda environment file at `{conda_file_path}` is not a valid file.')

        args = shlex.split(f'conda env update --name base -f {conda_file_path}')
        subprocess.call(args)

    # install pip environment
    if pip_file_path is not None:
        pip_file_path = Path(pip_file_path).absolute()
        if not pip_file_path.is_file():
            raise FileNotFoundError(f'Pip requirement file at `{pip_file_path}` is not a vliad file.')

        args = shlex.split(f'python -m pip install -U -r {pip_file_path} --no-index')
        subprocess.call(args)


def pre_install(config):
    """Execute installation script before the service running."""
    commands = config.pre_install
    if commands is None:
        return

    if not isinstance(commands, list):
        commands = [commands]

    for command in commands:
        args = shlex.split(command)
        subprocess.call(args)


def clean_up():
    conda_root = os.environ['CONDA_ROOT']

    commands = [
        'conda clean -ya',
        'rm -rf ~/.cache/pip',
        f'find {conda_root}/ -follow -type f -name "*.a" -delete',
        f'find {conda_root}/ -follow -type f -name "*.pyc" -delete',
        f'find {conda_root}/ -follow -type f -name "*.js.map" -delete',
        f'find {conda_root}/lib/python*/site-packages/bokeh/server/static -follow '
        '-type f -name "*.js" ! -name "*.min.js" -delete',
        'apt-get autoremove -y',
        'rm -rf /var/lib/apt/lists/*'
    ]

    for command in commands:
        args = shlex.split(command)
        subprocess.call(args)


if __name__ == '__main__':
    config_env = load_config().env

    # change to third plugged-in directory
    print('Switch to /content/third')
    Path('/content/third').mkdir(parents=True, exist_ok=True)
    os.chdir('/content/third')

    fix_eol()
    print('Installing environment...')
    install_environment(config_env)
    print('Running customer scripts...')
    pre_install(config_env)
    print('Cleaning...')
    clean_up()
    print('Finish!')
