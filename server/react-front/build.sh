#!/usr/bin/env bash
# parameter 1: name of rendered html (e.g. index.html)

base_dir=${PWD}
react_dir=./react-front/
static_dir=./static/
template_dir=./templates/

cd ${react_dir}
# build static files
npm run-script build
cd ${base_dir}

## change js path
python fix_js_path.py ${react_dir}/build/ $1

# create a copy of build static files
mkdir tmp
cp -r ${react_dir}build/* tmp/

# move static folder to static common
mv tmp/*html ${template_dir}
mv tmp/* ${static_dir}
cp -rfl ${static_dir}static/* ${static_dir}
rm -r ${static_dir}static/

# clear temp
rm -r tmp
