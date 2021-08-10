import os
import smart_open
import json
import re
import urllib.request


def load_http_text(url):
    with urllib.request.urlopen(url) as f:
        return f.read().decode("utf-8")


def load_text(path):
    if path.startswith("http://") or path.startswith("https://"):
        return load_http_text(path)
    else:
        with smart_open.open(path) as f:
            return f.read()


def load_json(path):
    return json.loads(load_text(path))


def load_json_lines(path):
    return [json.loads(line) for line in load_text(path).split("\n") if line]


def dump_json(data, path):
    with smart_open.open(path, "w") as f:
        json.dump(data, f)


def dump_dataframe(df, path):
    with smart_open.open(path, "w") as f:
        df.to_csv(f, index=False)


def ensure_path_exists(path):
    if "://" in path:
        # Buckets like GS/S3 don't need to pre-create the prefix/folder
        return

    if not os.path.exists(path):
        os.makedirs(path)


def word_count(text):
    # Count words in text, this isn't well-defined but we count regex full words and
    # single-char non-words (e.g. punctuation), similar to word tokenizers
    return len(re.findall(r"\w+|[^\w\s]", text))
