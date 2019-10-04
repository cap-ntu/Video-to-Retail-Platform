#!/usr/bin/env bash

cd hysia/server || return 1

python manage.py runserver 0.0.0.0:8000
