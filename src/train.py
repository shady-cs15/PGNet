from model import PGNet
from batcher import Batcher
import tensorflow as tf
from preprocessing import Vocab
from collections import *
import numpy as np
import config as cfg
import os

# build model
pg = PGNet()
pg.build_graph()

# set up optimizer
optimizer = tf.train.AdagradOptimizer(cfg.LEARNING_RATE, initial_accumulator_value=0.1)
gradients, vars = zip(*optimizer.compute_gradients(pg.loss_final))
gradients, _ = tf.clip_by_global_norm(gradients, 2.25)
train_op = optimizer.apply_gradients(zip(gradients, vars), global_step=pg.global_step)
print 'optimizer set up'

# setup batcher
vocab = Vocab(cfg.VOCAB_PATH, cfg.VOCAB_SIZE)
batcher = Batcher(cfg.DATA_PATH, vocab, False)

# model dir to save ckpt
if not os.path.exists('../models'):
    os.makedirs('../models')

saver = tf.train.Saver()
with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(cfg.TRAIN_STEPS):
            batch = batcher.next()
            init_cov = np.zeros([cfg.BATCH_SIZE, batch.enc_batch.shape[1]])
            
            _, loss = sess.run([train_op, pg.loss_final], feed_dict={
                                        pg.encoder_input : batch.enc_batch,
                                        pg.encoder_masks : batch.enc_padding_mask,
                                        pg.encoder_input_length : batch.enc_lens,
                                        pg.decoder_input : batch.dec_batch,
                                        pg.decoder_masks : batch.dec_padding_mask,
                                        pg.decoder_gt : batch.target_batch,
                                        pg.num_encoder_oovs : batch.max_art_oovs,
                                        pg.encoder_oov_idx : batch.enc_batch_extend_vocab,
                                        pg.coverage : init_cov
                                    })
            if (i+1) % cfg.DISP_STEP == 0:
                print loss

        # save ckpt
        save_path = saver.save(sess, "../models/model_final.ckpt")
