# GPT
This repository is a simple and clean [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)  implementation in TensorFlow.


## Dependencies
- Python 3.9.16
- TensorFlow 2.12.0
- TensorFlow Text 2.12.1
- TensorFlow Datasets 4.9.2
- KerasNLP 0.5.2
- Datasets 2.13.0

## Usage
### Train
The model is trained by default on the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. Use `--model_dir=<model_dir>` to specify the model directory name.
```
python train.py --model_dir=<model_dir> 
```

Some other options:
- The `model.py` functions are compiled with XLA. To disable XLA, set `jit_compile=False`.

### Generate
Use `--model_dir=<model_dir>` and `--context=<context>` to specify the model directory name and context.
```
python generate.py --model_dir=openwt_512_d_512 --context="Hello, I'm a language model" --k=50 --temp=0.9
```
This generates the following:
```
Hello, I'm a language modeler and I'm a regular user of the web, so you know what I mean by the name of that. We're talking about the idea of "the" and we're talking about the idea of what "the" means.

We're already talking about the idea of "the" and then we'll talk about the idea of the "a" that we're doing. We'll be saying that it will be a good idea. We've certainly talked about, but it's about the idea of the "A" in terms of how it'll be viewed. The idea is what we're talking about is what we're saying.

First, what about the "A" in terms of when it would be a better idea?

First we have a better idea. As the first part of a project we'll be taking a look at what's more like our 'A' and what's in it.

Second, what we're already talking about is if you want to have fun with it, they're not trying to get it done and that's what the project is going to look like.

Finally, what we'd like to do is look at what the actual language we'll be doing, and the process of doing it. As I mentioned in the above, we're going to look at what's going to be said if a good "a" is enough.

Finally, what really means what we'll be doing is how we'll look at the code we did and see when a person wants to come and see what's in it. Since we'll be doing this before we even get to that point, we'll be performing our project along the way.

Finally, what we'll be doing is in the way we're performing our project, and also as a whole, we'll be doing what we're doing.

So, let's be honest - I'm not doing these things for a while, but we're doing just that. This is just something we've got to get done.

I've been doing this since last year. Now I'm using our best-known and most popular language, and they're the first part of the project they've been doing. They're all about being there, and I'll be doing something like this in the future. As soon as we are finished, we'll be doing some pretty more work.

You know we've been doing this before. But just what's

```

### Pretrained GPT-Mini 
To download and try pretrained GPT-Mini (openwt_512_d_512), run `demo.ipynb`. If you want to fine-tune GPT-Mini using the pretrained weights, you will need to modify the code in the `demo.ipynb` notebook or create a new notebook specifically for fine-tuning.

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
- The `model.py` functions are compiled with XLA

## Licence
MIT
