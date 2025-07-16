#!/bin/bash
# Install system dependencies for Prophet
apt-get update
apt-get install -y gfortran libopenblas-dev liblapack-dev

# Install Python packages
pip install -r requirements.txt 