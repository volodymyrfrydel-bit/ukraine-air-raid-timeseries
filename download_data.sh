#!/usr/bin/env bash
# Downloads the volunteer air raid siren dataset from the upstream
# GitHub repository (Vadimkin/ukrainian-air-raid-sirens-dataset).
# Run this once before using data_loader.py / app.py.

set -e

mkdir -p data/raw

curl -sL -o data/raw/volunteer_data_en.csv \
  "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/volunteer_data_en.csv"

echo "Downloaded data/raw/volunteer_data_en.csv"
wc -l data/raw/volunteer_data_en.csv
