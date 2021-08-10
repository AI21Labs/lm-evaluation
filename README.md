# LM Evaluation Test Suite
This repo contains code for running the evaluations and reproducing the results from the [Jurassic-1 Technical Paper](http://TODO), with current support for running the tasks through both the [AI21 Studio API](https://studio.ai21.com/) and [OpenAI's GPT3 API](https://beta.openai.com/).

## Citation
Please use the following bibtex entry:
```
@techreport{J1WhitePaper,
  author = {Lieber, Opher and Sharir, Or and Lenz, Barak and Shoham, Yoav},
  title = {Jurassic-1: Technical Details And Evaluation},
  institution = {AI21 Labs},
  year = 2021,
  month = aug,
}
```

## Installation
```
git clone https://github.com/AI21Labs/lm-evaluation.git
cd lm-evaluation
pip install -e .
```

## Usage
The entry point for running the evaluations is lm_evaluation/run_eval.py, which receives a list of tasks and models to run. 

The models argument should be in the form "provider/model_name" where provider can be "ai21" or "openai" and the model name is one of the providers supported models.

When running through one of the API models, set the your API key(s) using the environment variables AI21_STUDIO_API_KEY and OPENAI_API_KEY. Make sure to consider the costs and quota limits of the models you are running beforehand.

Examples:
```console
# Evaluate hellaswag and storycloze on j1-large
python -m lm_evaluation.run_eval --tasks hellaswag storycloze --models ai21/j1-large

# Evaluate all multiple-choice tasks on j1-jumbo
python -m lm_evaluation.run_eval --tasks all_mc --models ai21/j1-jumbo

# Evaluate all docprob tasks on curie and j1-large
python -m lm_evaluation.run_eval --tasks all_docprobs --models ai21/j1-large openai/curie

```

## Datasets
The repo currently support the zero-shot multiple-choice and document probability datasets reported in the [Jurassic-1 Technical Paper](http://TODO).

### Multiple Choice
Multiple choice datasets are formatted as described in the [GPT3 paper](https://arxiv.org/abs/2005.14165), and the default reported evaluation metrics are those described there.

All our formatted datasets except for storycloze are publically available and referenced in [lm_evaluation/tasks_config.py](lm_evaluation/tasks_config.py). Storycloze needs to be [manually downloaded](https://cs.rochester.edu/nlp/rocstories/) and formatted, and the location should be configured through the environment variable 'STORYCLOZE_TEST_PATH'.

### Document Probabilities
Document probability tasks include documents from 19 data sources, including [C4](https://www.tensorflow.org/datasets/catalog/c4) and datasets from ['The Pile'](https://arxiv.org/abs/2101.00027).

Each document is pre-split at sentence boundaries to sub-documents of up to 1024 GPT tokens each, to ensure all models see the same inputs/contexts regardless of tokenization, and to support evaluation of models which are limited to sequence lengths of 1024.

Each of the 19 tasks have ~4MB of total text data.

## Additional Configuration

### Results Folder
By default all results will be saved to the folder 'results', and rerunning the same tasks will load the existing results. The results folder can be changed using the environment variable LM_EVALUATION_RESULTS_DIR.
