import os
import re
from argparse import ArgumentParser

parser = ArgumentParser(description='Set input html file')
parser.add_argument('react_build_dir', nargs=1, help='react build directory')
parser.add_argument('html', nargs=1, default='index.html', help='html file name')
args = parser.parse_args()

FILE_PATH = os.path.join(*args.react_build_dir, *args.html)

line = ''
with open(FILE_PATH, 'r') as html:
    line = re.sub(r'/manifest.json', r'/static/manifest.json', html.read(), flags=re.M)
    line = re.sub(r'/favicon.ico', r'/static/favicon.icon', line, flags=re.M)

with open(FILE_PATH, 'w') as html:
    html.write(line)
