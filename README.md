# GPT
This repository is a simple and clean [GPT](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)  implementation in Tensorflow.


## Dependencies
- Python 3.8
- TensorFow 2.8
- TensorFlow Text 2.8.1
- TensorFlow Datasets 4.8.1
- Datasets 2.8.0

## Usage
### Train
The model trains by default on the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. Use `--model_dir=<model_dir>` to provide the model directory name.
```
python train.py --model_dir=<model_dir> 
```

Some other options:
- Use `--build_vocab=True` to build a WordPiece vocabulary.

### Generate
Use `--model_dir=<model_dir>` and `--context=<context>` to provide the model directory name and context.
```
python generate.py --model_dir=<model_dir> --context=<context>
```

### Hparams setting
Adjust hyperparameters on the `config.py` file.

### Tensorboard
Run `tensorboard --logdir ./`.


## References
- [Improving language understanding by generative pre-training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Language models are unsupervised multitask learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Language models are few-shot learners](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
- [minGPT](https://github.com/karpathy/minGPT)

Implementation notes:
- WordPiece tokenization


## Licence
MIT
