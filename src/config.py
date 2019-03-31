# hyperparameters

BATCH_SIZE = 16

VOCAB_SIZE = 50000 # update
EMB_DIM = 128
HID_DIM = 256

DECODER_MAX_STEPS = 100
ENCODER_MAX_STEPS = 400

# True if inference, False if train / val
INFERENCE = True
USE_COVERAGE = True
BEAM_SEARCH_K = 4

LEARNING_RATE = 0.15
TRAIN_STEPS = 200000
DISP_STEP = 200

DATA_PATH = '../data/chunks/train_*'
VOCAB_PATH = '../data/vocab'