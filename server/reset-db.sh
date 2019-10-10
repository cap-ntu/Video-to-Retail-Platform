#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"

# Reset database
rm -f "${DIR}"/db.sqlite3
rm -rf "${DIR}"/restapi/migrations

# Clean up files
rm -rf "${DIR}"/output

# create directory
# ├── audios
# ├── images
# │   ├── product
# │   │   ├── retrieved
# │   │   └── uploaded
# │   ├── retrieved
# │   ├── temp
# │   └── uploaded
# ├── json
# ├── multi_features
# ├── statistics
# ├── subtitles
# └── videos
mkdir -p "${DIR}"/output/{audios,images/{product/{retrieved,uploaded},retrieved,temp,uploaded},json,multi_features,statistics,subtitles,videos}

# Init database
python manage.py makemigrations restapi
python manage.py migrate
python manage.py loaddata dlmodels.json
# create super user with username=admin, password=admin
echo "Creating super user"
echo "from django.contrib.auth import get_user_model; User = get_user_model(); User.objects.create_superuser(
  'admin', 'admin@mhysia.com', 'admin')" | python manage.py shell
