# GPT
A simple implementation of [GPT (Generative Pre-trained Transformer)](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf).


## Dependencies
- Python 3.7
- Tensorfow 2.8
- TensorFlow Text 2.8.1


## Usage
### Train
Use `--file_pattern=<file_pattern>` to provide the dataset path and file pattern. Use `--context=<context>` to provide context.
```
python train.py --file_pattern=./dataset_path/*.txt --context=<context>
```

Some other options:
- Use `--build_vocab=True` to build a WordPiece vocabulary.

### Generate
Use `--model_dir=<model_dir>` and `--context=<context>` to provide the model directory name and context.
```
python generate.py --model_dir=<model_dir> --context=<context>
```

### Hparams setting
Adjust hyperparameters on the `hparams.py` file.

### Tensorboard
Run `tensorboard --logdir ./`.


## References

- [Improving language understanding by generative pre-training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Language models are unsupervised multitask learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Language models are few-shot learners](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
- [minGPT](https://github.com/karpathy/minGPT)

Implementation notes:
- WordPiece tokenization
- Cosine learning rate decay
- Clip gradients by global norm
- Gaussian Error Linear Unit (GELU) activation
- Adam with β1 = 0.9 and β2 = 0.95 


## Licence
MIT
