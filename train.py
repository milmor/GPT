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
from model import GPT
from utils import *
from config import config

AUTOTUNE = tf.data.experimental.AUTOTUNE


def create_ds(dataset, batch_size, min_seq_len, buffer_size=None):
    dataset = (
        dataset.filter(lambda x: tf.strings.length(x['text']) > min_seq_len)
    )
    dataset = (
        dataset.map(lambda x: tf_text.normalize_utf8(x['text'], 'NFKD'), 
                    num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
    )    
    if buffer_size:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        
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
	epochs = args.epochs
	ckpt_interval = args.ckpt_interval
	max_ckpt_to_keep = args.max_ckpt_to_keep
	context = args.context
	k = args.k
	print(config)
    
    # Dataset
	read_config = tfds.ReadConfig(
		shuffle_seed=config['shuffle_seed'],
	)

	raw_train_ds, raw_val_ds = tfds.load('wikipedia/20190301.en', 
								split=['train[:90%]', 'train[90%:]'],
								shuffle_files=True, read_config=read_config)
	print(f'\nTrain size: {len(raw_train_ds)} Val size: {len(raw_val_ds)}')
	
	raw_train_ds = create_ds(raw_train_ds, config['batch_size'], 
					config['min_seq_len'], config['buffer_size'])
	raw_val_ds = create_ds(raw_val_ds, config['batch_size'], 
					config['min_seq_len'])

	if build_vocab:
		build_vocabulary(raw_train_ds, vocab_size, vocab_file)
		
	tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
		vocabulary=config['vocab_file'],
		sequence_length=config['seq_len'] + 1,
		lowercase=False,
	)

	train_ds = raw_train_ds.map(lambda x: preprocess(x, tokenizer), 
			                    num_parallel_calls=tf.data.AUTOTUNE).prefetch(AUTOTUNE
	)

	val_ds = raw_val_ds.map(lambda x: preprocess(x, tokenizer), 
			                num_parallel_calls=tf.data.AUTOTUNE).prefetch(AUTOTUNE
	)

	# Model
	if config['decay_lr']:
		lr = tf.keras.optimizers.schedules.CosineDecay(config['learning_rate'], 
			                                           config['decay_steps'])
	else:
		lr = config['learning_rate']

	optimizer = tf.keras.optimizers.Adam(lr, 
			                             beta_1=config['beta_1'], 
			                             beta_2=config['beta_2'])
			                             
	model = GPT(optimizer, vocab_size=config['vocab_size'], 
		        maxlen=config['seq_len'], emb_dim=config['emb_dim'],
		        heads=config['heads'], mlp_dim=config['mlp_dim'],
		        depth=config['depth'], rate=config['rate'], 
		        initializer=config['initializer'])
		        
	# Checkpoint
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
		print('Checkpoint restored from {} at epoch {}'.format(ckpt_manager.latest_checkpoint,
		                                                       int(ckpt.epoch)))
		                                                       
	# Train
	start = int(ckpt.epoch) + 1
	for epoch in range(start, epochs):
		# Train step
		start = time.time()
		for inp, tar in train_ds:
		    model.train_step(inp, tar)
		
		print(f'\nTime taken for train epoch {epoch} is: {time.time() - start:.2f} secs')
		print(f'Train loss: {model.train_loss_avg.result():.4f}')
		
		# Test step
		for inp, tar in val_ds:
		    model.test_step(inp, tar)
		    
		print(f'Time taken for test epoch {epoch} is: {time.time() - start:.2f} secs')
		print(f'Val loss: {model.test_loss_avg.result():.4f}')
		    
		generated_text = sample(model, context, config['seq_len'],
							config['vocab_file'], k=k)
		print(f'Generated text:\n{generated_text}')
		
		# Tensorboard
		with writer.as_default():
		    tf.summary.scalar('train_loss', model.train_loss_avg.result(), step=epoch)
		    tf.summary.scalar('test_loss', model.test_loss_avg.result(), step=epoch)
		    
		model.train_loss_avg.reset_states()
		model.test_loss_avg.reset_states()
		
		# Checkpoint
		if epoch % ckpt_interval == 0:
		    ckpt_manager.save(epoch)
		    print(f'Checkpoint saved at epoch {epoch}\n') 
		    
		ckpt.epoch.assign_add(1)
																																												

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='wiki_en_model')
    parser.add_argument('--build_vocab', default=False)
    parser.add_argument('--epochs', default=100000)
    parser.add_argument('--ckpt_interval', type=int, default=5)
    parser.add_argument('--max_ckpt_to_keep', type=int, default=3)  
    parser.add_argument('--context', default='The world is')  
    parser.add_argument('--k', type=int, default=5)  
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
