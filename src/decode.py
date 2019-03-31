from model import PGNet
from batcher import Batcher
import tensorflow as tf
from preprocessing import Vocab
from collections import *
import numpy as np
import config as cfg
import os
import preprocessing
import time

assert cfg.INFERENCE, 'cfg.INFERENCE should be True'
assert cfg.BATCH_SIZE==1, 'batch size should be 1 for inference mode'

# build model
pg = PGNet()
pg.build_graph()

# setup batcher
vocab = Vocab(cfg.VOCAB_PATH, cfg.VOCAB_SIZE)
batcher = Batcher(cfg.DATA_PATH, vocab, True)

saver = tf.train.Saver()

# create folder from summaries
if not os.path.exists('../summs'):
    os.makedirs('../summs')
    os.makedirs('../summs/generated')


with tf.Session() as sess:
    saver.restore(sess, '../models/model_final.ckpt')
    print 'model restored ..'

    batch = batcher.next()
    file_id = 0
    
    print '='*80
    print 'writing summaries one by one ..'
    time.sleep(10)
    print 'starting ..'

    while batch:
        encoder_outputs, reduced_state = sess.run([pg.encoder_outputs, pg.reduced_state],
                                        feed_dict={
                                        pg.encoder_input : batch.enc_batch,
                                        pg.encoder_input_length : batch.enc_lens,
                                    })

        decoder_input_state = [reduced_state for k in range(cfg.BEAM_SEARCH_K)]
        running_probabilities = np.ones([cfg.BEAM_SEARCH_K], dtype=np.float32)
        running_summary = [''] * cfg.BEAM_SEARCH_K

        input_token = [vocab.word2id(preprocessing.START_DECODING) for k in range(cfg.BEAM_SEARCH_K)]
        init_cov = [np.zeros([cfg.BATCH_SIZE, batch.enc_batch.shape[1]]) for k in range(cfg.BEAM_SEARCH_K)]

        for t in range(cfg.DECODER_MAX_STEPS):
                extended_probs = []
                extended_ids = []
                running_dec_states = []
                running_cov = []
                for k in range(cfg.BEAM_SEARCH_K):
                        decoder_input = input_token[k] if input_token[k] >= cfg.VOCAB_SIZE else vocab.word2id(preprocessing.UNKNOWN_TOKEN) 
                        decoder_input = np.array([[decoder_input]])
                        input_dic = {
                                pg.reduced_state : decoder_input_state[k],
                                pg.decoder_input : decoder_input,
                                pg.encoder_outputs : encoder_outputs,
                                pg.encoder_masks : batch.enc_padding_mask,
                                pg.coverage : init_cov[k],
                                pg.num_encoder_oovs : batch.max_art_oovs,
                                pg.encoder_oov_idx : batch.enc_batch_extend_vocab
                        }

                        output_tensors = [
                                pg.p_attn,
                                pg.final_coverage,
                                pg.decoder_state_final,
                                pg.topk_token_ids,
                                pg.topk_probs
                        ]
                        attn, cov, decoder_state, ids, probs = sess.run(output_tensors, feed_dict=input_dic)

                        probs*= running_probabilities[k]
                        extended_probs += list(probs)
                        extended_ids += list(ids)

                        running_dec_states += [decoder_state]
                        running_cov += [cov]

                # update running probabilities
                topk_prob_inds = np.argsort(probs)[-cfg.BEAM_SEARCH_K:]
                k1 = topk_prob_inds % cfg.BEAM_SEARCH_K
                k2 = topk_prob_inds
                running_probabilities = probs[k2]
                
                # update tokens
                ids = ids[k2]
                words = preprocessing.outputids2words(ids, vocab, batch.art_oovs[0])
                input_token = [vocab.word2id(words[k]) for k in range(cfg.BEAM_SEARCH_K)]

                # update summaries
                running_summary_updt = [running_summary[k] for k in k1]
                running_summary = [running_summary_updt[k] + ' ' + words[k] for k in range(cfg.BEAM_SEARCH_K)]
                
                decoder_input_state = [running_dec_states[i] for i in k1]
                init_cov = [running_cov[i] for i in k1]
                
        
        # save summaries
        with open(os.path.join('../summs/generated', str(file_id)+'.txt'), 'w') as f:
                f.write(running_summary[0])
        
        print '\033[Fsummaries written: %d' %(file_id+1)
        batch = batcher.next()
        file_id += 1
