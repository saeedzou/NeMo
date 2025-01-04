#!/bin/bash
# Install necessary Python packages
echo "Installing Python packages..."
python -m pip install nemo_toolkit['asr']
python -m pip install wget mutagen sox
python -m pip install huggingface_hub
python -m pip install pytorch-lightning==2.2.1
python -m pip install lightning
wget "https://raw.githubusercontent.com/saeedzou/NeMo/main/tools/ctc_segmentation/requirements.txt" -O requirements.txt
python -m pip install -r requirements.txt
# Clone and install ParsNorm
git clone https://github.com/saeedzou/ParsNorm.git
cd ParsNorm && pip install -e . && pip install -r requirements.txt
echo "Installation completed."