#!/bin/bash

# Install system dependencies
brew install cmake

# Install Python dependencies
pip3 install -r requirements.txt
pip3 install dlib
pip3 install face-recognition
pip3 install mediapipe
pip3 install -r requirements-dev.txt

# Create test resources
mkdir -p tests/resources
python3 tests/create_test_resources.py

# Run tests
python3 -m unittest discover tests -v 