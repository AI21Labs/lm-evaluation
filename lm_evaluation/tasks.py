import numpy as np
from collections import defaultdict
from lm_evaluation.utils import word_count


def multiple_choice_evaluate(dataset, model):
    """Runs multiple-choice eval on the dataset. Each entry in the dataset is of the format:
    {
        "candidates": [{"context": str, "completion": str}*N]
        "target_index": int,
        "answer_context": str
    }

    Where target_index is the correct candidate index

    Returns a dictionary of metrics, including accuracy using various normalization methods (tokens, chars, words, and
    answer_context normalized if answer_context is provided), as well as percentage of requests that had unaligned
    completions (i.e. the completion started in the middle of a token)
    """

    reqs = []
    metrics = defaultdict(list)

    # Generate and send all the requests first
    for row in dataset:
        assert row['target_index'] in range(len(row['candidates']))
        for cand in row['candidates']:
            ans_context = row.get('answer_context', None)

            reqs.append({"context": cand['context'], "completion": cand['completion']})
            # If there's an answer context, calculate also the probability of the suffix conditioned only on the ans ctx
            if ans_context is not None:
                reqs.append({"context": ans_context, "completion": cand['completion']})

    results = model.batched_conditional_logprobs(reqs)

    # Process the results
    for row in dataset:
        preds = defaultdict(list)
        for i, cand in enumerate(row['candidates']):
            cand_pred = results.pop(0)
            probs = cand_pred['logprobs']
            # Track how many requests had unaligned completions
            metrics['unaligned'].append(cand_pred['completion'] != cand_pred['aligned_completion'])

            # Calculate acc with token/char/no normalizing of the suffix, as well as answer-context normalized
            # probablities if there's an answer context supplied
            if row.get('answer_context', None) is not None:
                cand_pred_ans_ctx = results.pop(0)
                ans_probs = cand_pred_ans_ctx['logprobs']
                # Answer-context normalized logprobs
                preds['acc_norm_ans_ctx'].append(np.sum(probs) - np.sum(ans_probs))

            prob_sum = np.sum(probs)
            preds['acc_no_norm'].append(prob_sum)
            preds['acc_norm_tokens'].append(prob_sum / len(probs))
            preds['acc_norm_chars'].append(prob_sum / len(cand['completion']))
            preds['acc_norm_words'].append(prob_sum / word_count(cand['completion']))

        for k, v in preds.items():
            metrics[k].append(np.argmax(v) == row['target_index'])

    # Make sure we consumed all the requests we sent
    assert not results

    summary = {
        k: np.mean(v) for k, v in metrics.items()
    }
    return summary


def doc_probs_evaluate(dataset, model):
    """Calculates byte-normalized logprobs of the given documents. Each entry in dataset should contain a 'text' field
    which is assumed to not exceed the max tokens of the model being used. Entries belonging to the same original
    document should have the same value in the 'doc_idx' field"""

    doc_probs = defaultdict(list)
    doc_lengths = defaultdict(int)

    reqs = [{"context": "", "completion": row['text']} for row in dataset]
    for row, result in zip(dataset, model.batched_conditional_logprobs(reqs)):
        doc_idx = row['doc_idx']
        doc_probs[doc_idx].extend(result['logprobs'])
        doc_lengths[doc_idx] += len(row['text'].encode("utf-8"))

    # Return average of the per-doc average, and general/flat average. Currently these are very similar for our
    # datasets since the eval docs have similar length, but might be different otherwise
    return {
        "doc_logprob_per_byte": np.mean([np.sum(doc_probs[doc_idx]) / doc_lengths[doc_idx] for doc_idx in doc_probs]),
        "logprob_per_byte": np.sum(sum(doc_probs.values(), [])) / np.sum(list(doc_lengths.values()))
    }
