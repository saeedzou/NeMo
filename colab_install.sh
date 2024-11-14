#!/bin/bash
# Set the branch and the NeMo directory path
BRANCH="main"
NEMO_DIR_PATH="NeMo"
# Install necessary Python packages
echo "Installing Python packages..."
python -m pip install nemo_toolkit['asr']
python -m pip install wget
python -m pip install sox
python -m pip install huggingface_hub==0.23.2
python -m pip install pytorch-lightning==2.2.1
wget "https://raw.githubusercontent.com/saeedzou/NeMo/main/tools/ctc_segmentation/requirements.txt" -O requirements.txt
python -m pip install -r requirements.txt
echo "Installation completed."