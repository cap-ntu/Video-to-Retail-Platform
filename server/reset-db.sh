#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" 2>/dev/null && pwd )"

# Reset database
rm -f $DIR/db.sqlite3
rm -r $DIR/restapi/migrations

# Clean up files
rm -r $DIR/output/json
mkdir $DIR/output/json
touch $DIR/output/json/.save

rm -r $DIR/output/videos
mkdir $DIR/output/videos
touch $DIR/output/videos/.save

rm -r $DIR/output/statistics
mkdir $DIR/output/statistics
touch $DIR/output/statistics/.save

rm -r $DIR/output/audios
mkdir $DIR/output/audios
touch $DIR/output/audios/.save

rm -r $DIR/output/subtitles
mkdir $DIR/output/subtitles
touch $DIR/output/subtitles/.save

rm -r $DIR/output/tables
mkdir $DIR/output/tables
touch $DIR/output/tables/.save

rm -r $DIR/output/images
mkdir $DIR/output/images
mkdir $DIR/output/images/product
mkdir $DIR/output/images/product/retrieved
touch $DIR/output/images/product/retrieved/.save
mkdir $DIR/output/images/product/uploaded
touch $DIR/output/images/product/uploaded/.save
mkdir $DIR/output/images/retrieved
touch $DIR/output/images/retrieved/.save
mkdir $DIR/output/images/uploaded
touch $DIR/output/images/uploaded/.save
mkdir $DIR/output/images/temp
touch $DIR/output/images/temp/.save

rm -r $DIR/output/multi_features
mkdir $DIR/output/multi_features
touch $DIR/output/multi_features/.save

# Init database
python manage.py makemigrations restapi
python manage.py migrate
python manage.py loaddata dlmodels.json
python manage.py createsuperuser
