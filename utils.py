'''GPT model for Tensorflow.

Author: Emilio Morales (mil.mor.mor@gmail.com)
        Mar 2022
'''
import glob
import time
import tensorflow as tf
import tensorflow_text as tf_text
import keras_nlp


def sample(model, context, seq_len, vocab_file, k=10):  
    # No padding tokenizer
    sample_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
        vocabulary=vocab_file,
        sequence_length=None,
        lowercase=False
    )
    x = sample_tokenizer(tf_text.normalize_utf8(context, 'NFKD'))
    x = x[tf.newaxis, :]
    context_len = x.shape[1]

    for i in range(context_len, seq_len):
        x_pad = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=seq_len, padding="post")

        logits = model(x_pad, training=False)

        logits, indices = tf.math.top_k(logits[:, i -1, :], k=k)
        logits = tf.keras.activations.softmax(logits)
        rand_idx = tf.random.categorical(logits, num_samples=1, dtype=tf.int64)
     
        sample = tf.cast(indices[0][rand_idx[0][0]], tf.int32)[tf.newaxis, tf.newaxis]
        x = tf.concat([x, sample], axis=-1)

    try: 
        out_text = sample_tokenizer.detokenize(x).numpy()[0].decode('utf-8') 
    except:
        print("utf-8' codec can't decode byte")
    return out_text


def write_vocab_file(vocab_file, vocab):
    with open(vocab_file, 'w') as f:
        for token in vocab:
            print(token, file=f)
