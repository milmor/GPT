config = {'batch_size': 16,
          'buffer_size': 40000, # shuffle buffer size
          'shuffle_seed': 32, # train / val split seed
          'min_seq_len': False,
          'ckpt_interval': 2000, # 2000,
          'val_steps': 1000, #1000,
          'train_size': 95, # 95% train / 5% val,
          # hparams
          'vocab_size': 50257, # gpt-2 vocab size
          'seq_len': 512,
          'learning_rate': 0.001,
          'beta_1': 0.9,
          'beta_2': 0.95,
          'decay_lr': False, # whether to decay the learning rate
          'decay_steps': 400000,
          'alpha': 0.1, # minimum learning rate value as a fraction of initial_learning_rate
          'emb_dim': 512,
          'heads': 8,
          'mlp_dim': 512,
          'depth': 10,
          'dropout':  0.0,
          'initializer': 'glorot_uniform',
          'embedding_initializer': 'glorot_uniform', 
          'eps': 1e-6,
          'mlp_activation': 'gelu'}
