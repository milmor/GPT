'''
Author: Emilio Morales (mil.mor.mor@gmail.com)
        Mar 2022
'''
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable tensorflow debugging logs
import time
import tensorflow as tf
import tensorflow_text as tf_text
import keras_nlp
import tensorflow_datasets as tfds
import json
from model import GPT
from utils import *
from config import config

AUTOTUNE = tf.data.experimental.AUTOTUNE


def create_ds(dataset, batch_size, min_seq_len=False, buffer_size=None):
    if min_seq_len:
        dataset = (
            dataset.filter(lambda x: tf.strings.length(x['text']) > min_seq_len)
        )
    dataset = (
        dataset.map(lambda x: tf_text.normalize_utf8(x['text'], 'NFKD'), 
                    num_parallel_calls=AUTOTUNE)
    )    
    if buffer_size:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = dataset.batch(batch_size)
    return dataset
    
    
def preprocess(inputs, tokenizer):
    tokenized_text = tokenizer(inputs)
    x = tokenized_text[:, :-1]
    y = tokenized_text[:, 1:]
    return x, y
    
    
def build_vocabulary(dataset, vocab_size, vocab_file):
    start = time.time()
    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
        dataset,
        vocabulary_size=vocab_size,
        lowercase=False,
        reserved_tokens=["[PAD]", "[UNK]", "[BOS]"],
    )

    print(f'Time for generate vocab is {time.time()-start:.4f} sec')
    write_vocab_file(vocab_file, vocab)
    print(f'{vocab_file} saved')


def train(args):
    print('\n#########')
    print('GPT Train')
    print('#########\n')
    model_dir = args.model_dir
    build_vocab = args.build_vocab
    steps = args.steps
    max_ckpt_to_keep = args.max_ckpt_to_keep
    context = args.context
    k = args.k
    ds_name = args.ds_name
    print(config)

    # Dataset
    read_config = tfds.ReadConfig(
        shuffle_seed=config['shuffle_seed'],
    )
    train_size = config['train_size']
    raw_train_ds, raw_val_ds = tfds.load(ds_name, 
                                split=[f'train[:{train_size}%]', 
                                       f'train[{train_size}%:]'],
                                shuffle_files=True, read_config=read_config)
    raw_val_ds = raw_val_ds.take(config['batch_size']*config['val_steps'])
    print(f'\nTrain size: {len(raw_train_ds)} Val size: {len(raw_val_ds)}')

    raw_train_ds = create_ds(raw_train_ds, config['batch_size'], 
                    config['min_seq_len'], config['buffer_size'])
    raw_val_ds = create_ds(raw_val_ds, config['batch_size'], 
                    config['min_seq_len'])

    if build_vocab:
        print('Creating vocabulary...')
        build_vocabulary(raw_train_ds.take(50000), 
        					config['vocab_size'], config['vocab_file'])

    tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
        vocabulary=config['vocab_file'],
        sequence_length=config['seq_len'] + 1,
        lowercase=False,
    )

    train_ds = raw_train_ds.map(lambda x: preprocess(x, tokenizer), 
                                num_parallel_calls=tf.data.AUTOTUNE).repeat().prefetch(
                                    AUTOTUNE
    )
    train_ds = iter(train_ds)
    test_input, _ = train_ds.get_next()

    val_ds = raw_val_ds.map(lambda x: preprocess(x, tokenizer), 
                            num_parallel_calls=tf.data.AUTOTUNE).prefetch(
                                AUTOTUNE
    )

    # Model
    if config['decay_lr']:
        lr = tf.keras.optimizers.schedules.CosineDecay(config['learning_rate'], 
                                                       config['decay_steps'],
                                                       config['alpha'])
    else:
        lr = config['learning_rate']

    optimizer = tf.keras.optimizers.Adam(lr, 
                        beta_1=config['beta_1'], 
                        beta_2=config['beta_2'])

    model = GPT(vocab_size=config['vocab_size'], 
                maxlen=config['seq_len'], emb_dim=config['emb_dim'],
                heads=config['heads'], mlp_dim=config['mlp_dim'],
                depth=config['depth'], rate=config['dropout'], 
                initializer=config['initializer'])

    model.compile(optimizer)
    model(test_input)
    model.summary()

    # Checkpoint
    log_dir = os.path.join(model_dir, 'log-dir')
    writer = tf.summary.create_file_writer(log_dir)

    checkpoint_dir = os.path.join(model_dir, 'training-checkpoints')
    best_checkpoint_dir = os.path.join(model_dir, 'best-training-checkpoints')
    ckpt = tf.train.Checkpoint(optimizer=optimizer,
                               model=model,
                               step=tf.Variable(0),
                               best_loss=tf.Variable(1000.0)) # initialize with big value

    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=checkpoint_dir, 
                                              max_to_keep=1)
    best_ckpt_manager = tf.train.CheckpointManager(ckpt, directory=best_checkpoint_dir, 
													max_to_keep=max_ckpt_to_keep)

    if ckpt_manager.latest_checkpoint:    
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Checkpoint restored from {} at step {}'.format(ckpt_manager.latest_checkpoint,
                                                               int(ckpt.step)))
    else:
        config_file = os.path.join(model_dir, model_dir + '_config.json')
        json_config = json.dumps(config)
        with open(config_file, 'w') as f:
            f.write(json_config)
        print(f'config {config_file} saved')

    # Train
    start_step = ckpt.step.numpy() + 1
    start = time.time()

    for step in range(start_step, steps):
        # Train loop
        inp, tar = train_ds.get_next()
        model.train_step(inp, tar)

        # Eval step
        if step % config['ckpt_interval'] == 0 and step >= config['ckpt_interval']:
            print(f'\nTime taken for step {step} is: {time.time() - start:.2f} secs')
            print(f'Train loss: {model.train_loss_avg.result():.4f}')
            if config['decay_lr']:
            	print(f'lr: {model.optimizer.learning_rate(step)}')

            # Val loop
            start = time.time()
            for inp, tar in val_ds:
                model.test_step(inp, tar)

            print(f'Time taken for validation is: {time.time() - start:.2f} secs')
            print(f'Val loss: {model.test_loss_avg.result():.4f}')

            generated_text = sample(model, 'The world is', 
                                config['seq_len'], config['vocab_file'], k=k)
            print(f'Generated text:\n{generated_text}')

            # Tensorboard
            with writer.as_default():
                tf.summary.scalar('train_loss', model.train_loss_avg.result(), step=step)
                tf.summary.scalar('val_loss', model.test_loss_avg.result(), step=step)
                if config['decay_lr']:
                	tf.summary.scalar('lr', model.optimizer.learning_rate(step), step=step)

            # Checkpoint
            if model.test_loss_avg.result() < ckpt.best_loss.numpy():
                ckpt.step.assign(step)
                ckpt.best_loss.assign(model.test_loss_avg.result())
                best_ckpt_manager.save(step)
                ckpt_manager.save(step)
                print(f'Best checkpoint saved at step {step}\n') 
            else:
                ckpt.step.assign(step)
                ckpt_manager.save(step)
                print(f'Checkpoint saved at step {step}\n')             

            model.train_loss_avg.reset_states()
            model.test_loss_avg.reset_states()
            start = time.time()  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='openwt_model')
    parser.add_argument('--build_vocab', default=False)
    parser.add_argument('--steps', type=int, default=1000000)   
    parser.add_argument('--max_ckpt_to_keep', type=int, default=3)  
    parser.add_argument('--context', default="Hello, I'm a language model")  
    parser.add_argument('--k', type=int, default=5)  
    parser.add_argument('--ds_name', default='huggingface:openwebtext/plain_text')  
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()

