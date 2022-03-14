'''
Author: Emilio Morales (mil.mor.mor@gmail.com)
        Mar 2022
'''
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable tensorflow debugging logs
import time
import tensorflow as tf
import tensorflow_text as text
from model import GPT
from utils import *
from hparams import hparams

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_file(filename):
    raw_text = tf.io.read_file(filename)
    return raw_text


def preprocess(raw_text, maxlen, vocab_file):
    tokenizer = text.BertTokenizer(vocab_file)
    tokenized_text = tokenizer.tokenize(raw_text).merge_dims(1, -1) 
    trimmer = text.RoundRobinTrimmer(max_seq_length=maxlen + 1)
    trimmed_feat = trimmer.trim([tokenized_text])
    input_word_ids, _ = text.pad_model_inputs(input=trimmed_feat[0], max_seq_length=maxlen + 1)
    x = input_word_ids[:, :-1]
    y = input_word_ids[:, 1:]
    return x, y


def create_ds(file_pattern, batch_size, maxlen, vocab_file):
    text_paths = tf.data.Dataset.list_files(file_pattern)
    BUFFER_SIZE = tf.data.experimental.cardinality(text_paths)
    print(f'{BUFFER_SIZE} files')
    text_paths = text_paths.cache().shuffle(BUFFER_SIZE)

    dataset = text_paths.map(load_file, 
                  num_parallel_calls=AUTOTUNE).batch(batch_size, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(lambda filename: preprocess(filename, maxlen, vocab_file), 
                          num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return dataset


def train(args):
    print('\n#########')
    print('GPT Train')
    print('#########\n')
    file_pattern = args.file_pattern
    model_dir = args.model_dir
    vocab_file = args.vocab_file
    build_vocab = args.build_vocab
    epochs = args.epochs
    ckpt_interval = args.ckpt_interval
    max_ckpt_to_keep = args.max_ckpt_to_keep
    context = args.context

    model = GPT(vocab_size=hparams['vocab_size'], 
                maxlen=hparams['maxlen'], emb_dim=hparams['emb_dim'],
                heads=hparams['heads'], mlp_dim=hparams['mlp_dim'],
                depth=hparams['depth'], rate=hparams['rate'], 
                initializer=hparams['initializer'])

    optimizer = tf.keras.optimizers.Adam(hparams['learning_rate'], 
                                         beta_1=hparams['beta_1'], 
                                         beta_2=hparams['beta_2'])

    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if build_vocab:
        generate_vocab(file_pattern, vocab_size, vocab_file)

    dataset = create_ds(file_pattern, hparams['batch_size'], hparams['maxlen'], vocab_file)
    tokenizer = text.BertTokenizer(vocab_file)

    log_dir = os.path.join(model_dir, 'log-dir')
    writer = tf.summary.create_file_writer(log_dir)

    checkpoint_dir = os.path.join(model_dir, 'training-checkpoints')
    ckpt = tf.train.Checkpoint(optimizer=optimizer,
                               model=model,
                               epoch=tf.Variable(0))

    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=checkpoint_dir, 
                                              max_to_keep=max_ckpt_to_keep)

    if ckpt_manager.latest_checkpoint:    
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print(f'Checkpoint restored from {ckpt_manager.latest_checkpoint} at epoch {int(ckpt.epoch)}')

    train_loss_avg = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def train_step(inp, tar):
        with tf.GradientTape() as tape:
            predictions = model(inp, training=True)
            loss = loss_function(tar, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss_avg(loss)

    for epoch in range(epochs):
        start = time.time()
        for inp, tar in dataset:
            train_step(inp, tar)
        
        print(f'\nTime taken for epoch {epoch} is: {time.time() - start:.2f} secs')
        print(f'Loss: {train_loss_avg.result():.4f}')
        generated_text = sample(model, context, hparams['maxlen'], tokenizer)
        print(f'Generated text:\n{generated_text}')
        
        with writer.as_default():
            tf.summary.scalar('train_loss', train_loss_avg.result(), step=epoch)
            
        train_loss_avg.reset_states()
        
        if epoch % ckpt_interval == 0:
            ckpt_manager.save(epoch)
            print(f'Checkpoint saved at epoch {epoch}\n') 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_pattern')
    parser.add_argument('--model_dir', default='model-1')
    parser.add_argument('--vocab_file', default='vocab.txt')
    parser.add_argument('--build_vocab', default=False)
    parser.add_argument('--epochs', type=int, default=10000)  
    parser.add_argument('--ckpt_interval', type=int, default=5)
    parser.add_argument('--max_ckpt_to_keep', type=int, default=3)  
    parser.add_argument('--context', default='Enter context here...')  
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
