'''GPT model for Tensorflow.

Author: Emilio Morales (mil.mor.mor@gmail.com)
        Mar 2022
'''
import glob
import time
import tensorflow as tf
import tensorflow_text as text
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

AUTOTUNE = tf.data.experimental.AUTOTUNE


def sample(model, context, maxlen, tokenizer, k=10):  
    tokenized_text = tokenizer.tokenize(context).merge_dims(1, -1) 
    context_len = tokenized_text[0].shape[0]
    trimmer = text.RoundRobinTrimmer(max_seq_length=maxlen + 1)
    trimmed_feat = trimmer.trim([tokenized_text])
    x = trimmed_feat[0]
    
    for i in range(context_len - 1, maxlen):
        x_pad, _ = text.pad_model_inputs(input=x, max_seq_length=maxlen)
       
        logits = model(x_pad, training=False)
   
        logits, indices = tf.math.top_k(logits[:, i, :], k=k)
        logits = tf.keras.activations.softmax(logits)
        rand_idx = tf.random.categorical(logits, num_samples=1, dtype=tf.int64)
     
        sample = tf.cast(indices[0][rand_idx[0][0]], tf.int64)[tf.newaxis, tf.newaxis]

        x = tf.concat([x, sample], axis=-1)

    str_list = tokenizer.detokenize(x).numpy()[0] 
    out_text = ' '.join([token.decode('utf-8') for token in str_list])
   
    return out_text

def build_vocabulary(file_pattern, vocab_size, vocab_file='vocab.txt', batch_size=128):
    filenames = glob.glob(file_pattern)
    text_ds = tf.data.TextLineDataset(filenames)
    
    bert_tokenizer_params=dict(lower_case=False)
    reserved_tokens=['[PAD]', '[UNK]']

    bert_vocab_args = dict(
        vocab_size = vocab_size,
        reserved_tokens=reserved_tokens,
        bert_tokenizer_params=bert_tokenizer_params
    )
    
    print(f'Building vocabulary from {len(filenames)} files...')
    start = time.time()
    vocab = bert_vocab.bert_vocab_from_dataset(
    text_ds.batch(batch_size, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE),
    **bert_vocab_args
    )
    
    write_vocab_file(vocab_file, vocab)
    print(f'{vocab_file} saved')
    print(f'Time for generate vocab is {time.time()-start} sec')

def write_vocab_file(vocab_file, vocab):
    with open(vocab_file, 'w') as f:
        for token in vocab:
            print(token, file=f)
