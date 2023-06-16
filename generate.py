'''
Author: Emilio Morales (mil.mor.mor@gmail.com)
        Mar 2022
'''
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable tensorflow debugging logs
import tensorflow as tf
import tensorflow_text as text
import json
from model import GPT
from utils import *


def generate(args):
    print('\n#############')
    print('GPT Generate')
    print('#############\n')
    model_dir = args.model_dir
    context = args.context
    max_len = args.max_len
    k = args.k
    
    # Load config file
    config_file = model_dir + "/" + model_dir + "_config.json"
    with open(config_file) as f:
    	config = json.load(f)
    	print(f'{config_file} restored')

    model = GPT(vocab_size=config['vocab_size'], 
                maxlen=config['seq_len'], emb_dim=config['emb_dim'],
                heads=config['heads'], mlp_dim=config['mlp_dim'],
                depth=config['depth'], rate=config['dropout'], 
                initializer=config['initializer'])
                
    tokenizer = keras_nlp.models.GPT2Tokenizer.from_preset("gpt2_base_en", 
					sequence_length=config['seq_len'])
    checkpoint_dir = os.path.join(model_dir, 'best-ckpt')
    ckpt = tf.train.Checkpoint(model=model,
                               step=tf.Variable(0)) # initialize with big value

    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=checkpoint_dir, 
                                              max_to_keep=1)

    if ckpt_manager.latest_checkpoint:    
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print(f'Checkpoint restored from {ckpt_manager.latest_checkpoint} at step {int(ckpt.step)}')

    generated_text = sample(model, context, config['seq_len'], max_len, k=k)
    print(f'\nGenerated text:\n{generated_text}')

    with open('generate.txt', 'w') as f:
        f.write(generated_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='openwt_model')
    parser.add_argument('--context', default="Hello, I'm a language model")  
    parser.add_argument('--max_len', type=int, default=512)  
    parser.add_argument('--k', type=int, default=10)  
    args = parser.parse_args()

    generate(args)


if __name__ == '__main__':
    main()
