#!/bin/bash

module load git
module load ffmpeg
module load python/3.9.2

nvidia-smi

git clone https://github.com/ostris/ai-toolkit
mkdir -p content/dataset

cd ai-toolkit && git submodule update --init --recursive && pip install -r requirements.txt

