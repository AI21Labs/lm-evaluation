import os

# Base path for our datasets. Note that storycloze path needs to be manually supplied
MC_BASE_PATH = "https://storage.googleapis.com/ai21-public-data/lm_evaluation/datasets/multiple_choice"

# Base path for our doc_prob datasets
DOC_PROBS_BASE_PATH = "https://storage.googleapis.com/ai21-public-data/lm_evaluation/datasets/doc_probs/max_seq_len_1024-4096KB"

# By default, this metric will be used in multiple-choice tasks. For ARC and RACE the answer-context
# normalized logprobs metric will be used as per the GPT3 paper
MC_DEFAULT_METRIC = "acc_norm_tokens"

_TASKS_CONFIG = {
    "arc-challenge": {
        "type": "multiple_choice",
        "test_dataset": f"{MC_BASE_PATH}/arc-challenge/test.jsonl",
        "main_metric": "acc_norm_ans_ctx"
    },
    "arc-easy": {
        "type": "multiple_choice",
        "test_dataset": f"{MC_BASE_PATH}/arc-easy/test.jsonl",
        "main_metric": "acc_norm_ans_ctx"
    },
    "hellaswag": {
        "type": "multiple_choice",
        "test_dataset": f"{MC_BASE_PATH}/hellaswag/validation.jsonl",
        "main_metric": MC_DEFAULT_METRIC
    },
    "piqa": {
        "type": "multiple_choice",
        "test_dataset": f"{MC_BASE_PATH}/piqa/validation.jsonl",
        "main_metric": MC_DEFAULT_METRIC
    },
    "race-high": {
        "type": "multiple_choice",
        "test_dataset": f"{MC_BASE_PATH}/race-high/test.jsonl",
        "main_metric": "acc_norm_ans_ctx"
    },
    "race-middle": {
        "type": "multiple_choice",
        "test_dataset": f"{MC_BASE_PATH}/race-middle/test.jsonl",
        "main_metric": "acc_norm_ans_ctx"
    },
    "storycloze": {
        "type": "multiple_choice",
        "test_dataset": os.environ.get("STORYCLOZE_TEST_PATH", None),
        "main_metric": MC_DEFAULT_METRIC
    },
    "winogrande": {
        "type": "multiple_choice",
        "test_dataset": f"{MC_BASE_PATH}/winogrande/validation.jsonl",
        "main_metric": MC_DEFAULT_METRIC
    },
    "rte": {
        "type": "multiple_choice",
        "test_dataset": f"{MC_BASE_PATH}/rte/validation.jsonl",
        "main_metric": MC_DEFAULT_METRIC
    },
    "boolq": {
        "type": "multiple_choice",
        "test_dataset": f"{MC_BASE_PATH}/boolq/validation.jsonl",
        "main_metric": MC_DEFAULT_METRIC
    }
}

# Add doc-prob tasks
_TASKS_CONFIG.update({
    name: {
        "type": "doc_probs",
        "test_dataset": f"{DOC_PROBS_BASE_PATH}/{name}.jsonl",
        "main_metric": "doc_logprob_per_byte"

    } for name in [
        "arxiv",
        "books3",
        "c4",
        "dm_math",
        "enron_emails",
        "freelaw",
        "github",
        "gutenberg",
        "hackernews",
        "nih_exporter",
        "open_subtitles",
        "phil_papers",
        "pile_cc",
        "pubmed_abstracts",
        "pubmed_central",
        "stackexchange",
        "ubuntu_irc",
        "uspto",
        "youtube_subtitles"
    ]
})


def get_task_config(task_name):
    assert task_name in _TASKS_CONFIG, f"No task '{task_name}'"
    return _TASKS_CONFIG[task_name]


def get_all_tasks_of_type(task_type):
    return [task_name for task_name, task_config in _TASKS_CONFIG.items() if task_config['type'] == task_type]
