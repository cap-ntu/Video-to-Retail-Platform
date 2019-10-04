#!/usr/bin/env bash

base_dir=${PWD}
react_dir=../react-front/
static_dir=../static/
template_dir=../templates/

cd ${react_dir} || return 1

# build static files
npm run-script build
cd "${base_dir}" || return 1

## change js path
python fix_js_path.py ${react_dir}/build/

# create a copy of build static files
mkdir -p tmp
cp -r ${react_dir}build/* tmp/

# move static folder to static common
mv tmp/*html ${template_dir}
mv tmp/* ${static_dir}
cp -rfl ${static_dir}static/* ${static_dir}
rm -r ${static_dir}static/

# clear temp
rm -r tmp
