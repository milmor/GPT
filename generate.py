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
from hparams import hparams
from utils import *


def generate(args):
    print('\n############')
    print('GPT Generate')
    print('############\n')
    model_dir = args.model_dir
    vocab_file = args.vocab_file
    context = args.context

    model = GPT(vocab_size=hparams['vocab_size'], 
                maxlen=hparams['maxlen'], emb_dim=hparams['emb_dim'],
                heads=hparams['heads'], mlp_dim=hparams['mlp_dim'],
                depth=hparams['depth'], rate=hparams['rate'], 
                initializer=hparams['initializer'])

    tokenizer = text.BertTokenizer(vocab_file)
    
    checkpoint_dir = os.path.join(model_dir, 'training-checkpoints')
    ckpt = tf.train.Checkpoint(model=model,
                               epoch=tf.Variable(0))
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=checkpoint_dir, 
                                              max_to_keep=1)

    if ckpt_manager.latest_checkpoint:    
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print(f'Checkpoint restored from {ckpt_manager.latest_checkpoint} at epoch {int(ckpt.epoch)}')

    generated_text = sample(model, context, hparams['maxlen'], tokenizer)

    with open('generate.txt', 'w') as f:
        f.write(generated_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='model-1')
    parser.add_argument('--vocab_file', default='vocab.txt')
    parser.add_argument('--context', default='Enter context here...') 
    args = parser.parse_args()

    generate(args)


if __name__ == '__main__':
    main()
