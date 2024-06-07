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
from config import config


def generate(args, conf):
    print('\n#############')
    print('GPT Generate')
    print('#############\n')
    model_dir = args.model_dir
    context = args.context
    max_len = args.max_len
    k = args.k
    
    # Load config file
    model = GPT(vocab_size=conf.vocab_size, 
                seq_len=conf.seq_len, emb_dim=conf.emb_dim,
                heads=conf.heads, mlp_dim=conf.mlp_dim,
                depth=conf.depth, rate=conf.dropout, 
                initializer=conf.initializer)
                
    tokenizer = keras_nlp.models.GPT2Tokenizer.from_preset("gpt2_base_en", 
					sequence_length=conf.seq_len)
    ckpt_dir = os.path.join(model_dir, 'best-ckpt')

    model.restore(ckpt_dir)
    generated_text = sample(model, context, max_len, k=k)
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
    conf = Config(config, args.model_dir)
    generate(args, conf)


if __name__ == '__main__':
    main()