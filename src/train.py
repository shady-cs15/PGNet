from model import PGNet
from mybatcher import Batcher
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
batcher = Batcher(vocab)

# model dir to save ckpt
if not os.path.exists('../models'):
    os.makedirs('../models')

saver = tf.train.Saver()
with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(cfg.TRAIN_STEPS):
            batch = batcher.next()
            init_cov = np.zeros([cfg.BATCH_SIZE, batch['encoder_input'].shape[1]])
            
            _, loss = sess.run([train_op, pg.loss_final], feed_dict={
                                        pg.encoder_input : batch['encoder_input'],
                                        pg.encoder_masks : batch['encoder_masks'],
                                        pg.encoder_input_length : batch['encoder_input_length'],
                                        pg.decoder_input : batch['decoder_input'],
                                        pg.decoder_masks : batch['decoder_masks'],
                                        pg.decoder_gt : batch['decoder_gt'],
                                        pg.num_encoder_oovs : batch['num_encoder_oovs'],
                                        pg.encoder_oov_idx : batch['encoder_oov_idx'],
                                        pg.coverage : init_cov
                                    })
            if i % cfg.DISP_STEP == 0:
                print loss

        # save ckpt
        save_path = saver.save(sess, "../models/model_final.ckpt")
