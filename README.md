# Code for "CodeMark: Imperceptible Watermarking for Code Datasets against Neural Code Completion Models"

This repo contains the source code introduced in our paper, including the scripts to reproduce our experiments and the tool demo CodeMarker.

## Download the dataset

CodeSearchNet dataset can be downloaded from [here](https://huggingface.co/datasets/code_search_net)

## Install dependencies
You can use ```requirements.txt``` to install the dependencies.

```bash
pip install -r requirements.txt
```

Tree-sitter needs to build its parser for each language. We have pre-built two parsers for Java and Python in '''build''' folder. If you want to use other languages, you need to build the parser by yourself. Refer to this link for more details: [tree-sitter](https://github.com/tree-sitter/py-tree-sitter). 


## Usage

### 1. Embed watermarks into code datasets using CodeMarker

You can read the code and comments in ```marker.py``` to understand how CodeMarker works.

### 2. Train/Verify watermarked models

Please refer to ```run.py``` for more details.

In this experiment, we download the Huggingface pretrained models into a local folder since the download speed is too slow. You can download the models from Huggingface and specify their paths in args ```cache_path```. If not doing so, you need to modify the ```args.cache_path``` in this repository to the name of the model.


