
import Queue
from random import shuffle
import time
import numpy as np
import tensorflow as tf
import preprocessing as ppp
from preprocessing import Example, text_generator
import config as cfg


class Batcher(object):
    def __init__(self, vocab):
        self.next_batch = None
        self.xy_generator = text_generator(ppp.example_generator(cfg.DATA_PATH, cfg.INFERENCE))
        self.vocab = vocab
        self.pad_id = vocab.word2id(ppp.PAD_TOKEN)

    def _update_next_batch(self):
        batch = []
        for bs in range(cfg.BATCH_SIZE):
            source, ref = self.xy_generator.next() 
            ref_sents = [sent.strip() for sent in ppp.abstract2sents(ref)]
            batch.append(Example(source, ref_sents, self.vocab))

        # sort by encoder input length and tokenize
        batch = sorted(batch, key=lambda x: x.enc_len)
        self._tokenize_batch(batch)


    def _tokenize_batch(self, batch):
        # pad encoder seqs and decoder seqs
        max_enc_len = max([x.enc_len for x in batch])
        for x in batch:
            x.pad_encoder_input(max_enc_len, self.pad_id)
            x.pad_decoder_inp_targ(cfg.DECODER_MAX_STEPS, self.pad_id)

        # intialize         
        encoder_input = np.zeros((cfg.BATCH_SIZE, max_enc_len), dtype=np.int32)
        encoder_input_length = np.zeros((cfg.BATCH_SIZE), dtype=np.int32)
        encoder_masks = np.zeros((cfg.BATCH_SIZE, max_enc_len), dtype=np.float32)
        decoder_input = np.zeros((cfg.BATCH_SIZE, cfg.DECODER_MAX_STEPS), dtype=np.int32)
        decoder_gt = np.zeros((cfg.BATCH_SIZE, cfg.DECODER_MAX_STEPS), dtype=np.int32)
        decoder_masks = np.zeros((cfg.BATCH_SIZE, cfg.DECODER_MAX_STEPS), dtype=np.float32)
        encoder_oov_idx = np.zeros((cfg.BATCH_SIZE, max_enc_len), dtype=np.int32)
        num_encoder_oovs = 0
        oovs = []
        
        # fill up
        for i, x in enumerate(batch):
            encoder_input[i, :], decoder_input[i, :], decoder_gt[i, :] = x.enc_input, x.dec_input[:], x.target
            num_encoder_oovs = max(num_encoder_oovs, len(x.article_oovs))
            oovs.append(x.article_oovs)
            for j in xrange(x.enc_len):
                encoder_masks[i][j] = 1
                if j < x.dec_len:
                    decoder_masks[i][j] = 1
            encoder_oov_idx[i, :] = x.enc_input_extend_vocab

        
        self.next_batch = {
            'encoder_input': encoder_input,
            'encoder_input_length': encoder_input_length,
            'encoder_masks': encoder_masks,
            'decoder_input': decoder_input,
            'decoder_gt': decoder_gt,
            'decoder_masks': decoder_masks,
            'encoder_oov_idx': encoder_oov_idx,
            'num_encoder_oovs': num_encoder_oovs,
            'oovs': oovs
        }


    def next(self):
        self._update_next_batch()
        return self.next_batch