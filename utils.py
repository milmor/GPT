'''GPT model for Tensorflow.

Author: Emilio Morales (mil.mor.mor@gmail.com)
        Mar 2022
'''
import tensorflow as tf
import tensorflow_text as tf_text
import keras_nlp


def sample(model, context, seq_len, max_len, k=10):
    # Initialize tokenizer
    sample_tokenizer = keras_nlp.models.GPT2Tokenizer.from_preset("gpt2_base_en")

    # Tokenize the given context
    x = sample_tokenizer.tokenize(tf_text.normalize_utf8(context, 'NFKD'))
    x = tf.expand_dims(x, 0)

    # Generate new text by sampling from the model
    for i in range(x.shape[1], max_len):
        # Pad the input sequence to seq_len
        x_pad = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=seq_len, padding="post")
        # Generate logits from the model
        logits = model(x_pad, training=False)
        # Get the top k predictions and their probabilities
        logits, indices = tf.math.top_k(logits[:, i-1, :], k=k, sorted=False)
        # Sample from the predicted probabilities
        rand_idx = tf.random.categorical(tf.math.log(logits), num_samples=1, dtype=tf.int32)
        sample = tf.gather_nd(indices, rand_idx, batch_dims=1)
        sample = tf.expand_dims(sample, 0)
        # Concatenate the new token to the sequence
        x = tf.concat([x, sample], axis=-1)

    # Detokenize the generated sequence
    out_text = sample_tokenizer.detokenize(x).numpy()[0].decode('utf-8', errors='replace') 
    return out_text


def write_vocab_file(vocab_file, vocab):
    with open(vocab_file, 'w') as f:
        for token in vocab:
            print(token, file=f)
