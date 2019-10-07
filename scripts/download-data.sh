#!/usr/bin/env bash

filenames=(hysia-decoder-lib-linux-x86-64.tar.gz weights.tar.gz object-detection-data.tar.gz)
fileids=(1fi-MSLLsJ4ALeoIP4ZjUQv9DODc1Ha6O 1O1-QT8HJRL1hHfkRqprIw24ahiEMkfrX 1an7KGVer6WC3Xt2yUTATCznVyoSZSlJG)
unzip_paths=(hysia/core/HysiaDecode . third/object_detection)

for i in $(seq 0 2);
do
  cd "${unzip_paths[i]}" || return 1
  echo "Download ${filenames[i]}"
  # Download
  curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileids[i]}" > /dev/null
  curl -Lb ./cookie \
      "https://drive.google.com/uc?export=download&confirm=$(awk '/download/ {print $NF}' ./cookie)&id=${fileids[i]}" \
      -o "${filenames[i]}"
  rm cookie
  # Unzip
  tar xvzf "${filenames[i]}" 2> /dev/null
  rm -f "${filenames[i]}"
  # Go back
  cd - || return 1
done
