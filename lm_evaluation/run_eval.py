import argparse
import os
import pandas as pd
from collections import defaultdict

from lm_evaluation.tasks_config import get_all_tasks_of_type, get_task_config
from lm_evaluation.utils import load_json_lines, ensure_path_exists, dump_json, dump_dataframe, load_json
from lm_evaluation.tasks import multiple_choice_evaluate, doc_probs_evaluate
from lm_evaluation.model_providers import make_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks',
                        type=str,
                        required=True,
                        nargs="+",
                        help="List of tasks to run, or 'all_mc'/'all_docprobs' for all multiple-choice/docprob tasks.")
    parser.add_argument('--models',
                        type=str,
                        nargs="+",
                        help='Models to run. Model is of the form provider/model, for example openai/davinci',
                        required=True)

    return parser.parse_args()


def run_evaluation(task, model):
    conf = get_task_config(task)
    dataset = load_json_lines(conf['test_dataset'])
    task_type = conf['type']
    if task_type == "multiple_choice":
        return multiple_choice_evaluate(dataset, model)
    elif task_type == "doc_probs":
        return doc_probs_evaluate(dataset, model)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def main():
    args = get_args()
    results_dir = os.environ.get("LM_EVALUATION_RESULTS_DIR", "results")
    ensure_path_exists(results_dir)

    tasks = args.tasks

    if tasks == ["all_mc"]:
        tasks = get_all_tasks_of_type("multiple_choice")
    elif tasks == ["all_docprobs"]:
        tasks = get_all_tasks_of_type("doc_probs")

    results = defaultdict(dict)
    for model_name in args.models:
        provider, model_name = model_name.split("/")
        model = make_model(provider, model_name)
        for task in tasks:
            results_file = os.path.join(results_dir, f"{provider}.{model_name}.{task}.json")
            try:
                # Load existing result if found
                result = load_json(results_file)
            except:
                result = None
            if result is None:
                print(f"Running {task} on {model_name}")
                result = run_evaluation(task, model)
                dump_json(result, results_file)
            print(f"{provider}/{model_name}/{task}/{result}")
            results[task][model_name] = result[get_task_config(task)["main_metric"]]

    # Summarize
    df = pd.DataFrame([{"task": k, "metric": get_task_config(k)["main_metric"], **v} for k, v in results.items()])
    df.loc['-'] = ["", "average"] + [df[model].mean() for model in df.columns[2:]]
    print(df)

    dump_dataframe(df, os.path.join(results_dir, "summary.csv"))


if __name__ == "__main__":
    main()