'''
Author: Emilio Morales (mil.mor.mor@gmail.com)
        Mar 2022
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable tensorflow debugging logs
import argparse
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


def create_ds(dataset, batch_size, buffer_size=None):
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

def train(args, conf):
    print('\n#############')
    print('GPT Train')
    print('#############\n')
    model_dir = args.model_dir
    steps = args.steps
    max_ckpt_to_keep = args.max_ckpt_to_keep
    context = args.context
    max_len = args.max_len
    k = args.k
    temp = args.temp
    ds_name = args.ds_name

    # Dataset
    read_config = tfds.ReadConfig(
        shuffle_seed=conf.shuffle_seed,
    )
    train_size = conf.train_size
    raw_train_ds, raw_val_ds = tfds.load(ds_name, 
                                split=[f'train[:{train_size}%]', 
                                       f'train[{train_size}%:]'],
                                shuffle_files=True, read_config=read_config)
    raw_val_ds = raw_val_ds.take(conf.batch_size * conf.val_steps)
    print(f'\nTrain size: {len(raw_train_ds)} Val size: {len(raw_val_ds)}')

    raw_train_ds = create_ds(raw_train_ds, conf.batch_size, conf.buffer_size)
    raw_val_ds = create_ds(raw_val_ds, conf.batch_size)

    tokenizer = keras_nlp.models.GPT2Tokenizer.from_preset("gpt2_base_en", 
			sequence_length=conf.seq_len + 1)

    train_ds = raw_train_ds.map(lambda x: preprocess(x, tokenizer), 
                num_parallel_calls=tf.data.AUTOTUNE).repeat().prefetch(
                                    AUTOTUNE
    )
    train_ds = iter(train_ds)
    val_ds = raw_val_ds.map(lambda x: preprocess(x, tokenizer), 
                            num_parallel_calls=tf.data.AUTOTUNE).prefetch(
                                AUTOTUNE
    )

    # Model
    if conf.decay_lr:
        lr = tf.keras.optimizers.schedules.CosineDecay(conf.learning_rate, 
                                                       conf.decay_steps,
                                                       conf.alpha)
    else:
        lr = conf.learning_rate

    optimizer = tf.keras.optimizers.Adam(lr, 
                        beta_1=conf.beta_1, 
                        beta_2=conf.beta_2)

    model = GPT(vocab_size=conf.vocab_size, 
                seq_len=conf.seq_len, emb_dim=conf.emb_dim,
                heads=conf.heads, mlp_dim=conf.mlp_dim,
                depth=conf.depth, rate=conf.dropout, 
                initializer=conf.initializer)

    model.compile(optimizer)
    model.summary()

    # Checkpoint
    log_dir = os.path.join(model_dir, 'log-dir')
    writer = tf.summary.create_file_writer(log_dir)

    checkpoint_dir = os.path.join(model_dir, 'ckpt')
    best_checkpoint_dir = os.path.join(model_dir, 'best-ckpt')
    ckpt = tf.train.Checkpoint(optimizer=model.optimizer,
                               model=model,
                               step=tf.Variable(0),
                               best_loss=tf.Variable(1000.0)) # initialize with big value

    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=checkpoint_dir, 
                                              max_to_keep=1)
    best_ckpt_manager = tf.train.CheckpointManager(ckpt, directory=best_checkpoint_dir, 
    							max_to_keep=max_ckpt_to_keep)

    if ckpt_manager.latest_checkpoint:    
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print(f'Checkpoint restored from {ckpt_manager.latest_checkpoint} at step {int(ckpt.step.numpy())}')
    else:
        print(f'New model')

    # Train
    start_step = ckpt.step.numpy() + 1
    start = time.time()
    
    # Train loop
    for step in range(start_step, steps):
        inp, tar = train_ds.get_next()
        model.train_step(inp, tar)

        # Eval step
        if step % conf.ckpt_interval == 0 and step >= conf.ckpt_interval:
            print(f'\nTime taken for step {step} is: {time.time() - start:.2f} secs')
            print(f'Train loss: {model.train_loss_avg.result():.4f}')
            if conf.decay_lr:
            	print(f'lr: {model.optimizer.learning_rate(step)}')

            # Val loop
            start = time.time()
            for inp, tar in val_ds:
                model.test_step(inp, tar)

            print(f'Time taken for validation is: {time.time() - start:.2f} secs')
            print(f'Val loss: {model.test_loss_avg.result():.4f}')

            generated_text = sample(model, context, max_len, k=k, temperature=temp)
            print(f'Generated text:\n{generated_text}')

            # Tensorboard
            with writer.as_default():
                tf.summary.scalar('train_loss', model.train_loss_avg.result(), step=step)
                tf.summary.scalar('val_loss', model.test_loss_avg.result(), step=step)
                if conf.decay_lr:
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
    parser.add_argument('--steps', type=int, default=1000000)   
    parser.add_argument('--max_ckpt_to_keep', type=int, default=3)  
    parser.add_argument('--context', default="Hello, I'm a language model")  
    parser.add_argument('--max_len', type=int, default=512)  
    parser.add_argument('--k', type=int, default=50)  
    parser.add_argument('--temp', type=float, default=0.9) 
    parser.add_argument('--ds_name', default='openwebtext/plain_text')  
    args = parser.parse_args()
    conf = Config(config, args.model_dir)
    train(args, conf)


if __name__ == '__main__':
    main()