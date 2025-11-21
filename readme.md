conda create --name improvnet python=3.11   
conda activate improvnet 
cd improvnet2/
pip install -e .

CUDA_VISIBLE_DEVICES=0,1,2,3 python improvnet/train/train.py