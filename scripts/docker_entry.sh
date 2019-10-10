#!/usr/bin/env bash
# Desc: Hysia docker entry point.
# Author: Zhou Shengsheng
# Date: 04-03-19

# Start model server and dashboard
cd server || return 1
python start_model_servers.py &
python manage.py runserver 0.0.0.0:8000
