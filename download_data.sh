#!/usr/bin/env bash
# Downloads the volunteer air raid siren dataset from the upstream
# GitHub repository (Vadimkin/ukrainian-air-raid-sirens-dataset).
#
# Note: app.py and src/data_loader.py now download this file
# automatically on first run if it's missing (see ensure_raw_data() in
# data_loader.py) -- this matters for cloud deployments like Streamlit
# Community Cloud, where there's no shell access to run this script
# before the app starts. Running this script manually is still useful
# for local development if you want the file pre-fetched, or want to
# force-refresh it without restarting the app.

set -e

mkdir -p data/raw

curl -sL -o data/raw/volunteer_data_en.csv \
  "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/volunteer_data_en.csv"

echo "Downloaded data/raw/volunteer_data_en.csv"
wc -l data/raw/volunteer_data_en.csv
