import tensorflow as tf 
import config as cfg 

class PGNet(object):
    def __init__(self):
        self.rand_inits = {'norm': tf.truncated_normal_initializer(stddev=1e-4),
                           'uni': tf.random_uniform_initializer(-0.02, 0.02),
                           'zeros': tf.zeros_initializer()
                          }

    def get_variables(self):
        # embedding matrix
        with tf.variable_scope('embedding'):
            emb_matrix = tf.get_variable('emb', [cfg.VOCAB_SIZE, cfg.EMB_DIM],\
                    dtype=tf.float32, initializer=self.rand_inits['norm'])
            
        # state reduce
        with tf.variable_scope('state_reduce'):
            state_reduce_w0 = tf.get_variable('w0', [2*cfg.HID_DIM, cfg.HID_DIM], dtype=tf.float32, \
                                        initializer=self.rand_inits['norm'])
            state_reduce_b0 = tf.get_variable('b0', [cfg.HID_DIM], dtype=tf.float32, \
                                        initializer=self.rand_inits['norm'])
            state_reduce_w1 = tf.get_variable('w1', [2*cfg.HID_DIM, cfg.HID_DIM], dtype=tf.float32, \
                                        initializer=self.rand_inits['norm'])
            state_reduce_b1 = tf.get_variable('b1', [cfg.HID_DIM], dtype=tf.float32, \
                                        initializer=self.rand_inits['norm'])

        # attention
        with tf.variable_scope('attention'):
            W_h = tf.get_variable('w_h', [1, 1, 2*cfg.HID_DIM, 2*cfg.HID_DIM], dtype=tf.float32, \
                                        initializer=self.rand_inits['norm'])
            W_s = tf.get_variable('w_s', [2*cfg.HID_DIM, 2*cfg.HID_DIM], dtype=tf.float32)
            b_attn = tf.get_variable('b_attn', [2*cfg.HID_DIM], dtype=tf.float32, \
                                        initializer=self.rand_inits['zeros'])
            #if cfg.USE_COVERAGE:
            W_c = tf.get_variable('w_c', [1, 1, 1, 2*cfg.HID_DIM], dtype=tf.float32, \
                                        initializer=self.rand_inits['norm'])
            V = tf.get_variable('v', [2*cfg.HID_DIM], dtype=tf.float32, initializer=self.rand_inits['norm'])
            
        # distribution
        with tf.variable_scope('distribution'):
            W_x = tf.get_variable('w_x', [2*cfg.HID_DIM + cfg.EMB_DIM, cfg.EMB_DIM], dtype=tf.float32)
            b_x = tf.get_variable('b_x', [cfg.EMB_DIM], dtype=tf.float32, \
                                        initializer=self.rand_inits['zeros'])
            W_s_dist = tf.get_variable('w_s', [4*cfg.HID_DIM + cfg.EMB_DIM, 1], dtype=tf.float32)
            b_ptr = tf.get_variable('b_ptr', [1], dtype=tf.float32, \
                                        initializer=self.rand_inits['zeros'])
            W_v0 = tf.get_variable('w_v0', [3*cfg.HID_DIM, cfg.HID_DIM], dtype=tf.float32)
            b_v0 = tf.get_variable('b_v0', [cfg.HID_DIM], dtype=tf.float32, \
                                        initializer=self.rand_inits['zeros'])
            W_v1 = tf.get_variable('w_v1', [cfg.HID_DIM, cfg.VOCAB_SIZE], dtype=tf.float32, \
                                        initializer=self.rand_inits['norm'])
            b_v1 = tf.get_variable('b_v1', [cfg.VOCAB_SIZE], dtype=tf.float32, \
                                        initializer=self.rand_inits['zeros'])

        return {'emb_matrix': emb_matrix,
                'state_reduce_w0': state_reduce_w0,
                'state_reduce_b0': state_reduce_b0,
                'state_reduce_w1': state_reduce_w1,
                'state_reduce_b1': state_reduce_b1,
                'attention_wh': W_h,
                'attention_wc': W_c,
                'attention_ws': W_s,
                'attention_b': b_attn,
                'attention_v': V,
                'dist_wx': W_x,
                'dist_bx': b_x,
                'dist_ws': W_s_dist,
                'dist_b': b_ptr,
                'dist_wv0': W_v0,
                'dist_bv0': b_v0,
                'dist_wv1': W_v1,
                'dist_bv1': b_v1}


    def build_graph(self):
        # setup placeholders
        self.encoder_input = tf.placeholder(tf.int32, [cfg.BATCH_SIZE, None])
        self.encoder_masks = tf.placeholder(tf.float32, [cfg.BATCH_SIZE, None])
        self.encoder_input_length = tf.placeholder(tf.int32, [cfg.BATCH_SIZE])
        decoder_steps = cfg.DECODER_MAX_STEPS
        if cfg.INFERENCE:
            decoder_steps = 1
        self.decoder_input = tf.placeholder(tf.int32, [cfg.BATCH_SIZE, decoder_steps])
        self.decoder_masks = tf.placeholder(tf.int32, [cfg.BATCH_SIZE, cfg.DECODER_MAX_STEPS])
        self.decoder_gt = tf.placeholder(tf.int32, [cfg.BATCH_SIZE, cfg.DECODER_MAX_STEPS])
        self.num_encoder_oovs = tf.placeholder(tf.int32, [])
        self.encoder_oov_idx = tf.placeholder(tf.int32, [cfg.BATCH_SIZE, None])
        self.coverage = tf.placeholder(tf.float32, [cfg.BATCH_SIZE, None])

        # setup extra variables
        variables = self.get_variables()
        with tf.variable_scope('global_step'):
            self.global_step = tf.Variable(0, trainable=False)

        print '='*80
        print 'tensor shapes '
        print '='*80
        
        # embedding lookup 
        with tf.variable_scope('embedding'):
            self.encoder_embeddings = tf.nn.embedding_lookup(variables['emb_matrix'], self.encoder_input)
            # TODO change
            #self.decoder_embeddings = [tf.nn.embedding_lookup(variables['emb_matrix'], x) for x in tf.unstack(self.decoder_input, axis=1)]
            self.decoder_embeddings = tf.nn.embedding_lookup(variables['emb_matrix'], self.decoder_input)
            print 'encoder embeddings shape:', self.encoder_embeddings.shape
            #print 'decoder embeddings shape: [' + str(len(self.decoder_embeddings)) + ',' +str(self.decoder_embeddings[0].shape)[1:]
            
        # build encoder
        with tf.variable_scope('encoder'):
            self.encoder_cell_f = tf.contrib.rnn.LSTMCell(cfg.HID_DIM, initializer=self.rand_inits['uni'])
            self.encoder_cell_b = tf.contrib.rnn.LSTMCell(cfg.HID_DIM, initializer=self.rand_inits['uni'])
            self.encoder_outputs, (self.encoder_state_f, self.encoder_state_b) = tf.nn.bidirectional_dynamic_rnn(self.encoder_cell_f,\
                        self.encoder_cell_b, self.encoder_embeddings, dtype=tf.float32, sequence_length=self.encoder_input_length)
            self.encoder_outputs = tf.concat(self.encoder_outputs, axis=-1)
            print 'encoder outputs shape:', self.encoder_outputs.shape
        
            # reduce encoder state
            self.encoder_state_0 = tf.concat([self.encoder_state_f[0], self.encoder_state_b[0]], axis=-1) 
            self.encoder_state_1 = tf.concat([self.encoder_state_f[1], self.encoder_state_b[1]], axis=-1)
            self.reduced_state_0 = tf.nn.relu(tf.matmul(self.encoder_state_0, variables['state_reduce_w0']) + variables['state_reduce_b0'])
            self.reduced_state_1 = tf.nn.relu(tf.matmul(self.encoder_state_1, variables['state_reduce_w1']) + variables['state_reduce_b1'])
            self.reduced_state = tf.contrib.rnn.LSTMStateTuple(self.reduced_state_0, self.reduced_state_1) 
        
        # build decoder
        with tf.variable_scope('decoder'):
            self.decoder_cell = tf.contrib.rnn.LSTMCell(cfg.HID_DIM, initializer=self.rand_inits['uni'])
            self.decoder_state = self.reduced_state
            
            # precompute W_h * h_i s for attention
            # use hack: conv2d with 1 convolution kernel
            self.encoder_outputs_transformed = tf.nn.conv2d(tf.expand_dims(self.encoder_outputs, 2), filter=variables['attention_wh'], \
                                                        strides=(1, 1, 1, 1), padding='SAME')
            print 'transformed encoder outputs shape:', self.encoder_outputs_transformed.shape

            

            if cfg.USE_COVERAGE:
                coverage = self.coverage

            # initialize context
            context_t = tf.zeros([cfg.BATCH_SIZE, 2*cfg.HID_DIM])
            
            self.losses = []
            # unroll decoder steps
            decoder_steps = cfg.DECODER_MAX_STEPS
            if cfg.INFERENCE:
                decoder_steps = 1
            for t in range(decoder_steps):
                if t>0:
                    tf.get_variable_scope().reuse_variables()

                # get x_t and update cell
                decoder_embedding = self.decoder_embeddings[:, t, :]
                #decoder_embedding = self.decoder_embeddings[t]
                x_t = tf.matmul(tf.concat((decoder_embedding, context_t), 1), variables['dist_wx']) + variables['dist_bx']
                decoder_output, self.decoder_state = self.decoder_cell(x_t, self.decoder_state)

                # Run attention, compute attn_t, coverage, update context_t
                # Computing attn_t
                # compute W_s * s_t for attn_t
                decoder_state_concat = tf.concat((self.decoder_state[0], self.decoder_state[1]), axis=1)
                decoder_state_transformed = tf.matmul(decoder_state_concat, variables['attention_ws']) + variables['attention_b']
                
                # compute coverage_feats
                if cfg.USE_COVERAGE:
                    coverage_transformed = tf.nn.conv2d(tf.expand_dims(tf.expand_dims(coverage, -1), -1), variables['attention_wc'], \
                                                            strides=(1, 1, 1, 1), padding='SAME')
                
                # compute attention distribution at time t
                linear_feats = tf.expand_dims(tf.expand_dims(decoder_state_transformed, axis=1), 1)\
                                    + self.encoder_outputs_transformed
                                    
                if cfg.USE_COVERAGE:
                    linear_feats += coverage_transformed
                
                
                                    
                attn_input_feat = variables['attention_v'] * tf.nn.tanh(linear_feats)
                if t == 0:
                    self.aif = attn_input_feat
                e_t = tf.reduce_sum(attn_input_feat, axis=[2, 3])
                attn_t = tf.nn.softmax(e_t) * self.encoder_masks
                attn_t /= tf.reshape(tf.reduce_sum(attn_t, 1), [-1, 1])
                if cfg.USE_COVERAGE and t > 0:
                    coverage += attn_t

                if t == 0:
                    print 'attention shape:', attn_t.shape
                    if cfg.USE_COVERAGE:
                        print 'coverage shape:', self.coverage.shape
                
                # compute coverage loss
                if t > 0 and cfg.USE_COVERAGE:
                    loss_cov = tf.reduce_sum(tf.minimum(attn_t, self.coverage), [1])
                    loss_cov *= tf.cast(mask, tf.float32)

                    
                # compute context vector: weighted sum of encoder states
                context_t = tf.reduce_sum(tf.expand_dims(tf.expand_dims(attn_t, -1), -1) * tf.expand_dims(self.encoder_outputs, 2)\
                                            ,  [1, 2])
                context_t = tf.reshape(context_t, [-1, 2*cfg.HID_DIM])

                # compute p_gen
                pgen_linear_feats = tf.matmul(tf.concat([context_t, decoder_state_concat, x_t], 1), variables['dist_ws']\
                                                            + variables['dist_b'])
                
                p_gen = tf.nn.sigmoid(pgen_linear_feats)
                if t==0:
                    self.p_gen = p_gen

                # compute p_vocab
                p_vocab_linear0 = tf.matmul(tf.concat([decoder_output, context_t], 1), variables['dist_wv0']) \
                                    + variables['dist_bv0']
                p_vocab_linear1 = tf.matmul(p_vocab_linear0, variables['dist_wv1']) \
                                    + variables['dist_bv1']
                                    
                p_vocab = tf.nn.softmax(p_vocab_linear1)

                # compute final distribution
                p_vocab *= p_gen
                p_vocab = tf.concat([p_vocab, tf.zeros([cfg.BATCH_SIZE, self.num_encoder_oovs])], 1)
                if t == 0:
                    self.p_attn = attn_t
                attn_t *= (1-p_gen)
                scatter_idxs = tf.stack([tf.tile(tf.expand_dims(tf.range(cfg.BATCH_SIZE), 1), [1, tf.shape(self.encoder_oov_idx)[1]]), \
                                            self.encoder_oov_idx], axis=2)
                p_attn = tf.scatter_nd(scatter_idxs, attn_t, [cfg.BATCH_SIZE, cfg.VOCAB_SIZE + self.num_encoder_oovs])
                p_final = p_attn + p_vocab

                # compute loss wrt output
                output = self.decoder_gt[:, t]
                p_gold = tf.gather_nd(p_final, tf.stack([tf.range(cfg.BATCH_SIZE), output], 1))
                loss_s2s = -tf.log(p_gold)
                if t == 0:
                    self.gp = p_gold
                
                # compute total loss, mask and add
                loss_total = loss_s2s
                if t > 0 and cfg.USE_COVERAGE: 
                    loss_total += loss_cov
                mask = self.decoder_masks[:, t]
                loss_total *= tf.cast(mask, tf.float32) 
                self.losses.append(loss_total)
            
            lengths = tf.cast(tf.reduce_sum(self.decoder_masks, axis=1), tf.float32)
            self.losses = sum(self.losses)/lengths    
            self.loss_final = tf.reduce_mean(self.losses)
            if cfg.USE_COVERAGE:
                self.final_coverage = coverage
            self.context_final = context_t
            self.p_final = p_final
            self.decoder_state_final = self.decoder_state

            if cfg.INFERENCE:
                topk_info = tf.nn.top_k(self.p_final[0], k=cfg.BEAM_SEARCH_K)
                self.topk_token_ids = topk_info[1]
                self.topk_probs = topk_info[0]

        print 'Graph generated ..'
        print '='*80
        print 'variables in memory ..'
        print '='*80
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for i, var in enumerate(variables):
            print i, var.name, ':', var.shape
        print '='*80
