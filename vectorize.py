#!/usr/bin/env python3

"""
Manages computing and storing embedding vectors for conversation turns
"""

import faiss
import json
import math
import numpy as np
from openai import OpenAI
import os
import re
import sys
import util

BATCH_SIZE = 1024
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

EMBEDDINGS_FILE = "snippets_with_embeddings.jsonl"

from sentence_transformers import SentenceTransformer

openai_client = OpenAI()


def chunks(reader, n):
    buf = []
    for row in reader:
        buf.append(row)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def corpus_directory(corpus):
    return os.path.join("indexes", corpus)


def index_data(use_local_embeddings, corpus="all"):
    entry_idx = 0
    num_rows = 0
    rows = None
    print("Building FAISS index...")
    corpus_dir = corpus_directory(corpus)
    fname = os.path.join(corpus_dir, EMBEDDINGS_FILE)
    with open(fname) as fs_in:
        for line in fs_in:
            num_rows += 1
    with open(fname) as fs_in:
        for line in fs_in:
            x = json.loads(line)
            vector = x["embedding"]
            vector_length = len(vector)
            if rows is None:
                rows = np.zeros((num_rows, vector_length), dtype="float32")
                quantizer = faiss.IndexFlatIP(vector_length)
                faiss_index = faiss.IndexIVFPQ(
                    quantizer,
                    vector_length,
                    int(math.sqrt(len(rows))),
                    vector_length // 2,  # (number of PQ segments)
                    4,
                )
            rows[entry_idx] = vector
            entry_idx += 1

    faiss_index.train(rows)
    faiss_index.add(rows)
    faiss.write_index(faiss_index, faiss_filename(use_local_embeddings, corpus))


def faiss_filename(use_local_embeddings, corpus):
    return os.path.join(
        corpus_directory(corpus),
        "fora_%s.faiss" % (use_local_embeddings and "local" or "openai"),
    )


def openai_encode(strings):
    postprocessed_strings = []
    for s in strings:
        s = s[:4000]  # max chunk size in OpenAI is 8191
        if not s:
            s = "none"  # It doesn't like empty strings
        postprocessed_strings.append(s)
    try:
        response = openai_client.embeddings.create(
            input=postprocessed_strings, model=OPENAI_EMBEDDING_MODEL
        )
    except:
        print(json.dumps(postprocessed_strings))
        raise
    embeddings = [x.embedding for x in response.data]
    return np.array(embeddings)


def local_encode():
    return SentenceTransformer(LOCAL_EMBEDDING_MODEL).encode


def get_faiss_index(use_local_embeddings=True, corpus="all"):
    filename = faiss_filename(use_local_embeddings, corpus)
    faiss_index = faiss.read_index(filename)
    faiss_index.nprobe = 50
    return faiss_index


def get_encoder(use_local_embeddings=True):
    if use_local_embeddings:
        return local_encode()
    else:
        return openai_encode


def add_vectors(use_local_embeddings, corpus="all"):
    encode = get_encoder(use_local_embeddings=use_local_embeddings)
    corpus_dir = corpus_directory(corpus)
    fname = os.path.join(corpus_dir, EMBEDDINGS_FILE)
    fs_out = open(fname, "w")
    j = 0
    for batch in chunks(open(os.path.join(corpus_dir, "snippets.jsonl")), BATCH_SIZE):
        inputs = [json.loads(line) for line in batch]
        strings = [x["content"] for x in inputs]
        embeddings = encode(strings)
        for i, obj in enumerate(inputs):
            obj["embedding"] = [round(a, 5) for a in np.array(embeddings[i]).tolist()]
            print(json.dumps(obj), file=fs_out)
            j += 1
        print("Encoded %s snippets." % (j))


if __name__ == "__main__":
    use_local_embeddings = True
    util.init_corpora()
    for corpus in util.CORPORA:
        print("Processing", corpus)
        add_vectors(use_local_embeddings, corpus=corpus)
        index_data(use_local_embeddings, corpus=corpus)
