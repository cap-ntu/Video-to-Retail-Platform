#!/usr/bin/env bash

cd hysia/server || return 1

python model_server/start_model_servers.py
