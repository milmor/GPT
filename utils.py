'''GPT model for Tensorflow.

Author: Emilio Morales (mil.mor.mor@gmail.com)
        Mar 2022
'''
import tensorflow as tf
import tensorflow_text as tf_text
import keras_nlp
from huggingface_hub import hf_hub_download
import json


def sample(model, context, max_len, k=10):
    # Initialize tokenizer
    sample_tokenizer = keras_nlp.models.GPT2Tokenizer.from_preset("gpt2_base_en")

    # Tokenize the given context
    x = sample_tokenizer.tokenize(tf_text.normalize_utf8(context, 'NFKD'))
    x = tf.expand_dims(x, 0)

    # Generate new text by sampling from the model
    for i in range(x.shape[1], max_len):
        # Pad the input sequence to seq_len
        x_pad = tf.keras.preprocessing.sequence.pad_sequences(x, 
            maxlen=model.seq_len, padding="post"
        )
        # Generate logits from the model
        logits = model(x_pad, training=False)
        # Get the top k predictions and their probabilities
        logits, indices = tf.math.top_k(logits[:, i-1, :], k=k, sorted=False)
        # Sample from the predicted probabilities
        rand_idx = tf.random.categorical(tf.math.log(logits), 
            num_samples=1, dtype=tf.int32
        )
        sample = tf.gather_nd(indices, rand_idx, batch_dims=1)
        sample = tf.expand_dims(sample, 0)
        # Concatenate the new token to the sequence
        x = tf.concat([x, sample], axis=-1)

    # Detokenize the generated sequence
    out_text = sample_tokenizer.detokenize(x).numpy()[0].decode('utf-8', errors='replace') 
    return out_text


class Loader():
    def __init__(self):
        pass
        
    def download(self, ckpt_dir):
        hf_hub_download(repo_id="milmor/gpt-mini", 
            filename=f"{ckpt_dir}/ckpt-1760000.data-00000-of-00001",
            local_dir='./'
        )

        hf_hub_download(repo_id="milmor/gpt-mini", 
            filename=f"{ckpt_dir}/ckpt-1760000.index",
            local_dir='./'
        )

        hf_hub_download(repo_id="milmor/gpt-mini", 
            filename=f"{ckpt_dir}/checkpoint",
            local_dir='./')

        config_file = hf_hub_download(repo_id="milmor/gpt-mini", 
            filename="openwt_512_d_512/openwt_512_d_512_config.json",
            local_dir='./'
        )

        with open(config_file) as f:
            self.config = json.load(f)