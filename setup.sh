#!/bin/sh
python3 -m venv venv
source venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt