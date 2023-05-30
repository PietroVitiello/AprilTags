#!/bin/sh

echo "Install requirements"
pip install wheel
pip install -r requirements.txt
echo ""

echo "Installing Segment-Anything from Meta"
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
echo ""

echo "Downloading Owl-ViT weights"
python OwlViT.py
echo ""