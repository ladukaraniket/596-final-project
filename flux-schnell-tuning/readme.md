# Setting up environment on USC CARC Discovery Cluster

## Request resources using the following command
```bash
salloc --partition=gpu --ntasks=1 --cpus-per-task=8 --gpus-per-task=a40:1 --time=04:30:00 --mem=64G
```

## Create a python virtual environment and activate
```bash
virtualenv tune

source tune/bin/activate
```

## Load the following modules
```bash
module load gcc/12.3.0
module load python/3.11.4
module load git
```

## Clone Hugging Face [ai-toolkit](https://github.com/ostris/ai-toolkit/) repository
```bash
git clone https://github.com/ostris/ai-toolkit

cd ai-toolkit && git submodule update --init --recursive
```

## Install dependencies
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip3 install -r requirements.txt
```

## Create dataset
- Create a dataset directory at the root of the project.
- Add min of 5 images along with accompanying captions in text files.
- Follow the format : image_name.png, image_name.txt

## Run the appropriate script (Tuning / Inference)
```bash
python tuning.py

python inference.py
```

