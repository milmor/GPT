'''GPT model for Tensorflow.

Author: Emilio Morales (mil.mor.mor@gmail.com)
        Mar 2022
'''
import tensorflow as tf
from tensorflow.keras import layers


class MultiHeadAttention(layers.Layer):
    def __init__(self, model_dim, n_heads, rate=0.1, initializer='glorot_uniform'):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.model_dim = model_dim

        assert model_dim % self.n_heads == 0

        self.head_dim = model_dim // self.n_heads

        self.wq = layers.Dense(model_dim, kernel_initializer=initializer)
        self.wk = layers.Dense(model_dim, kernel_initializer=initializer)
        self.wv = layers.Dense(model_dim, kernel_initializer=initializer)
        
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        
        self.wo = layers.Dense(model_dim, kernel_initializer=initializer)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  
        k = self.wk(k)  
        v = self.wv(v)  

        q = self.split_heads(q, batch_size) 
        k = self.split_heads(k, batch_size)  
        v = self.split_heads(v, batch_size) 

        dh = tf.cast(self.head_dim, tf.float32)
        qk = tf.matmul(q, k, transpose_b=True)
        scaled_qk =  qk / tf.math.sqrt(dh)
        
        if mask is not None:
            scaled_qk += (mask * -1e9) 

        attn = self.dropout1(tf.nn.softmax(scaled_qk, axis=-1))
        attn = tf.matmul(attn, v) 

        attn = tf.transpose(attn, perm=[0, 2, 1, 3]) 
        original_size_attention = tf.reshape(attn, (batch_size, -1, self.model_dim)) 

        output = self.dropout2(self.wo(original_size_attention))
        return output
    
    
class TransformerBlock(layers.Layer):
    def __init__(self, emb_dim, n_heads, mlp_dim, 
                 rate=0.1, initializer='glorot_uniform', eps=1e-6, activation='gelu'):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(emb_dim, n_heads, initializer=initializer)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation=activation, kernel_initializer=initializer), 
            layers.Dense(emb_dim, kernel_initializer=initializer),
            layers.Dropout(rate)
        ])
        self.ln1 = layers.LayerNormalization(epsilon=eps)
        self.ln2 = layers.LayerNormalization(epsilon=eps)

    def call(self, inputs, mask):
        x = self.ln1(inputs)
        x = inputs + self.attn(x, x, x, mask) 
        x = x + self.mlp(self.ln2(x))
        return x
    
    
class TokenEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, emb_dim, 
                 rate=0.1, initializer='glorot_uniform'):
        super(TokenEmbedding, self).__init__()
        self.max_len = maxlen
        self.token_emb = layers.Embedding(
            input_dim=vocab_size, output_dim=emb_dim, 
            embeddings_initializer=initializer)
        self.position_emb = layers.Embedding(
            input_dim=maxlen, output_dim=emb_dim, 
            embeddings_initializer=initializer)
        self.dropout = layers.Dropout(rate)

    def call(self, x):
        token_embeddings = self.token_emb(x)
        positions = tf.range(start=0, limit=self.max_len, delta=1)
        positions = self.position_emb(positions)
        return self.dropout(token_embeddings + positions) 
        
        
class GPT(tf.keras.models.Model):
    def __init__(self, vocab_size=50000, maxlen=512, 
                 emb_dim=256, heads=8, mlp_dim=256, depth=10, 
                 rate=0.1, initializer='glorot_uniform', 
                 embedding_initializer='glorot_uniform', eps=1e-6,
                 mlp_activation='gelu'):
        super(GPT, self).__init__()
        self.depth = depth
        self.tok_emb = TokenEmbedding(maxlen, vocab_size, 
                        emb_dim, rate=rate, initializer=embedding_initializer)
        self.drop = layers.Dropout(rate)
            
        self.transformer = [TransformerBlock(emb_dim, 
                                heads, mlp_dim, rate=rate,
                                initializer=initializer, eps=eps, 
                                activation=mlp_activation)
                            for _ in range(depth)]

        self.layernorm = layers.LayerNormalization(epsilon=eps)
        self.out = layers.Dense(vocab_size, kernel_initializer=initializer)
        
            
    def compile(self, optimizer):
        super(GPT, self).compile()
        self.optimizer = optimizer
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.train_loss_avg = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss_avg = tf.keras.metrics.Mean(name='val_loss')
        
    def get_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def get_attention_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)
    
    def create_mask(self, x):
        attn_mask = self.get_attention_mask(tf.shape(x)[1])
        padding_mask = self.get_padding_mask(x)
        attn_mask = tf.maximum(padding_mask, attn_mask)
        return attn_mask
                       
    def call(self, x):
        mask = self.create_mask(x)
 
        x = self.tok_emb(x)
        x = self.drop(x)

        for i in range(self.depth):
            x = self.transformer[i](x, mask)

        x = self.layernorm(x)
        x = self.out(x)
        return x
    
    @tf.function
    def train_step(self, inp, tar):
        with tf.GradientTape() as tape:
            predictions = self(inp, training=True)
            loss = self.loss_function(tar, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss_avg(loss)
        
    @tf.function
    def test_step(self, inp, tar):
        predictions = self(inp, training=False)
        loss = self.loss_function(tar, predictions)

        self.test_loss_avg(loss)
