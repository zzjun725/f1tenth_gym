import os, sys

BATCHSIZE = 32
# BATCHSIZE = 128
SEQ_DIM = 1
N_DIM = 2
EPOCHES = 100
L = 3
DROP_FIRST = 100
DT = 0.01

if len(sys.argv) > 3:
    SCENE = sys.argv[1] + '/'
    EXP_NAME = sys.argv[2]
    CUDA_NAME = sys.argv[3]
elif len(sys.argv) > 2:
    SCENE = sys.argv[1] + '/'
    EXP_NAME = sys.argv[2]
    CUDA_NAME = 'cuda'
else:
    print('Please give scene and exp name.')
DATA_DIR = 'data/'
CUDA_NAME = 'cpu'
# CUDA_NAME = 'cuda:0'

INSTABILITY_RECOVER = 1
USE_MIX_PRECISION_TRAINING = 0
MOVE_DATA_TO_DEVICE = 1

LR = 5e-4
