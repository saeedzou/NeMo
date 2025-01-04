#!/bin/bash
# Install necessary Python packages
echo "Installing Python packages..."
pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]
python -m pip install sox
wget "https://raw.githubusercontent.com/saeedzou/NeMo/main/tools/ctc_segmentation/requirements.txt" -O requirements.txt
python -m pip install -r requirements.txt
# Clone and install ParsNorm
git clone https://github.com/saeedzou/ParsNorm.git
cd ParsNorm && pip install -e . && pip install -r requirements.txt
python -m pip install hezar mutagen
echo "Installation completed."