# GPT
This repository is a simple and clean [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)  implementation in TensorFlow.


## Dependencies
- Python 3.11.2
- TensorFlow 2.12.0
- TensorFlow Text 2.12.1
- TensorFlow Datasets 4.9.2
- KerasNLP 0.4.1
- Datasets 2.11.0

## Usage
### Train
The model is trained by default on the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. Use `--model_dir=<model_dir>` to specify the model directory name.
```
python train.py --model_dir=<model_dir> 
```

Some other options:
- Use `--build_vocab=True` to build a WordPiece vocabulary.
- The `model.py` functions are compiled with XLA. To disable XLA, set `jit_compile=False`.

### Generate
Use `--model_dir=<model_dir>` and `--context=<context>` to specify the model directory name and context.
```
python generate.py --model_dir=<model_dir> --context=<context>
```

### Pretrained GPT-Mini 
To download and try pretrained GPT-Mini, run `demo.ipynb`. If you want to fine-tune GPT-Mini using the pretrained weights, you will need to modify the code in the `demo.ipynb` notebook or create a new notebook specifically for fine-tuning.

### Hparams setting
Adjust hyperparameters in the `config.py` file.

### Tensorboard
Run `tensorboard --logdir ./`.


## References
- [Improving language understanding by generative pre-training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Language models are unsupervised multitask learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Language models are few-shot learners](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
- [minGPT](https://github.com/karpathy/minGPT)

Implementation notes:
- WordPiece tokenization
- The `model.py` functions are compiled with XLA

## Licence
MIT
