'''GPT model for Tensorflow.

Author: Emilio Morales (mil.mor.mor@gmail.com)
        Mar 2022
'''
import os
import tensorflow as tf
import tensorflow_text as tf_text
import keras_nlp
from huggingface_hub import hf_hub_download
import json


def next_token(x, i, model, temperature, k):
    logits = model(x, training=False) / temperature
    logits, indices = tf.math.top_k(logits[:, i-1, :], k=k, sorted=False)
    probabilities = tf.nn.softmax(logits, axis=-1)
    rand_idx = tf.random.categorical(tf.math.log(probabilities), num_samples=1, dtype=tf.int32)
    sample = tf.gather_nd(indices, rand_idx, batch_dims=1)

    # Set the value at position i in x to sample
    x = tf.tensor_scatter_nd_update(x, [[0, i]], sample)
    return x

def sample(model, tokenizer, context, max_len, k=10, temperature=1.0, seed=None):
    x = tokenizer.tokenize(tf_text.normalize_utf8(context, 'NFKD'))
    x = tf.expand_dims(x, 0)
    start_len = x.shape[1]
    x = tf.keras.preprocessing.sequence.pad_sequences(x, 
        maxlen=model.seq_len, padding="post"
    )

    if seed is not None:
        tf.random.set_seed(seed)

    for i in range(start_len, max_len):
        x = next_token(x, i, model, temperature, k)
    
    # Find the first occurrence of zero and truncate the sequence at that point
    zero_positions = tf.where(tf.equal(x, 0))
    if tf.size(zero_positions) > 0:
        first_zero_pos = zero_positions[0, 1]  # Get the position of the first zero
        x = x[:, :first_zero_pos]

    out_text = tokenizer.detokenize(x).numpy()[0].decode('utf-8', errors='replace') 
    return out_text


class Config(object):
    def __init__(self, input_dict, save_dir):
        file_path = os.path.join(save_dir, f"{save_dir}_config.json")
        # Check if the configuration file exists
        if os.path.exists(file_path):
            self.load_config(file_path)
        else:
            for key, value in input_dict.items():
                setattr(self, key, value)
            self.save_config(file_path, save_dir)
            
        print(self.__dict__)

    def save_config(self, file_path, save_dir):
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Convert input_dict to JSON and save to file
        with open(file_path, "w") as f:
            json.dump(vars(self), f, indent=4)
        print(f'New config {file_path} saved')

    def load_config(self, file_path):
        # Load configuration from the existing file
        with open(file_path, "r") as f:
            config_data = json.load(f)

        # Update the object's attributes with loaded configuration
        for key, value in config_data.items():
            setattr(self, key, value)
        print(f'Config {file_path} loaded')

        
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