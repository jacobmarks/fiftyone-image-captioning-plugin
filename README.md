## Image Captioning Plugin

![image_captioning](https://github.com/jacobmarks/fiftyone-image-captioning-plugin/assets/12500356/224503c0-c3ac-4925-8c9d-ecfe50d493cc)


## Plugin Overview

This plugin lets you generate and store captions for your samples using
state-of-the-art image captioning models.

### Supported Models

This version of the plugin supports the following models:

- [BLIP Base](https://huggingface.co/Salesforce/blip-image-captioning-base) from Hugging Face
- [BLIPv2](https://replicate.com/andreasjansson/blip-2) (via [Replicate](https://replicate.com/))
- [Fuyu-8b](https://replicate.com/lucataco/fuyu-8b/) from Adept AI (via [Replicate](https://replicate.com/))
- [GiT](https://huggingface.co/docs/transformers/en/model_doc/git) from Hugging Face
- [Llava-1.5-7b](https://huggingface.co/llava-hf/llava-1.5-7b-hf) from Hugging Face
- [Llava-13b](https://replicate.com/yorickvp/llava-13b) (via [Replicate](https://replicate.com/))
- [Qwen-vl-chat](https://replicate.com/lucataco/qwen-vl-chat) (via [Replicate](https://replicate.com/)
- [ViT-GPT2](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) from Hugging Face

Feel free to fork this plugin and add support for other models!

## Installation

### Pre-requisites

1. If you plan to use it, install the Hugging Face transformers library:

```shell
pip install transformers
```

2. If you plan to use it, install the Replicate library:

```shell
pip install replicate
```

And add your Replicate API key to your environment:

```shell
export REPLICATE_API_TOKEN=<your-api-token>
```

### Install the plugin

```shell
fiftyone plugins download https://github.com/jacobmarks/fiftyone-image-captioning-plugin
```

## Operators

### `caption_images`

- Applies the selected image captioning model to the desired target view, and
  stores the resulting captions in the specified field on the samples.
