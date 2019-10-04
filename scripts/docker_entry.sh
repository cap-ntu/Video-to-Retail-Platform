#!/usr/bin/env bash
# Desc: Hysia docker entry point.
# Author: Zhou Shengsheng
# Date: 04-03-19

# Activate conda if installed
if [[ -f ${HOME}/anaconda3/bin/activate ]]; then
  source "${HOME}"/anaconda3/bin/activate && conda activate Hysia
fi

# Start model server and dashboard
bash start_model_server.sh &
bash run_dashboard.sh
