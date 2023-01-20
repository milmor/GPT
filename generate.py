'''
Author: Emilio Morales (mil.mor.mor@gmail.com)
        Mar 2022
'''
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable tensorflow debugging logs
import tensorflow as tf
import tensorflow_text as text
from model import GPT
from config import config
from utils import *


def generate(args):
    print('\n############')
    print('GPT Generate')
    print('############\n')
    model_dir = args.model_dir
    context = args.context
    k = args.k

    model = GPT(vocab_size=config['vocab_size'], 
                maxlen=config['seq_len'], emb_dim=config['emb_dim'],
                heads=config['heads'], mlp_dim=config['mlp_dim'],
                depth=config['depth'], rate=config['dropout'], 
                initializer=config['initializer'])
                
    tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
        vocabulary=config['vocab_file'],
        sequence_length=config['seq_len'] + 1,
        lowercase=False,
    )
    
    checkpoint_dir = os.path.join(model_dir, 'training-checkpoints')
    ckpt = tf.train.Checkpoint(model=model,
                               step=tf.Variable(0)) # initialize with big value

    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=checkpoint_dir, 
                                              max_to_keep=1)

    if ckpt_manager.latest_checkpoint:    
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print('Checkpoint restored from {} at step {}'.format(ckpt_manager.latest_checkpoint,
                                                               int(ckpt.step)))

    generated_text = sample(model, context, config['seq_len'],
                            config['vocab_file'], k=k)
    print(f'Generated text:\n{generated_text}')

    with open('generate.txt', 'w') as f:
        f.write(generated_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='openwt_model')
    parser.add_argument('--context', default='The world is')  
    parser.add_argument('--k', type=int, default=5)  
    args = parser.parse_args()

    generate(args)


if __name__ == '__main__':
    main()
