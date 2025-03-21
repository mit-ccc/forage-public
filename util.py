from collections import defaultdict
import json
import os
import re

import numpy as np
import pandas as pd

import gpt_lib
import vectorize

# If this is set to True, you will need a config.yml file in the home directory which provides the logins and
# hashed passwords of valid users
# (See https://blog.streamlit.io/streamlit-authenticator-part-1-adding-an-authentication-component-to-your-app/)
# If it's set to False, no authentication will be performed.
USE_AUTHENTICATION = False


def get_public_conversations():
    """Get the public Fora collection IDs, stored in a JSON file provided by Hope."""
    return json.loads(open("public_conversation_ids.json").read())


def get_catalog_202_conversations():
    """Get the conversations highlighted by catalog 202."""
    return json.loads(open("public_conversation_ids.json").read())


OBJECTIVES = {
    "generate_themes": {
        "display": "Generate themes about...",
        "prompt_template": "What are themes about %s in these remarks?",
    },
    "find_positive_mentions": {
        "display": "Find positive mentions of...",
        "prompt_template": "What are positives things people say about %s?",
    },
    "find_narratives": {
        "display": "What are the common narratives related to...",
        "prompt_template": "What are the narratives people have about %s?",
    },
    "find_negative_mentions": {
        "display": "Find negative mentions of...",
        "prompt_template": "What are negative things people say about %s?",
    },
    "summarize_opinions": {
        "display": "Summarize opinions about...",
        "prompt_template": "Please summarize the opinions people are expressing about %s.",
    },
    "summarize_experiences": {
        "display": "Summarize personal experiences shared about...",
        "prompt_template": "Please summarize the personal stories people are sharing about %s.",
    },
    "generate_pitches": {
        "display": "Generate story pitches about...",
        "prompt_template": 'Please generate some story pitches related to the topic of "%s" based on what people are expressing in these remarks.',
    },
    "best_comments": {
        "display": "Find the most memorable quotes about...",
        "prompt_template": 'Find the most memorable quotes about the topic of "%s" from these remarks.',
    },
    "other": {
        "display": "Other objective",
        "prompt_template": "%s",
    },
}


# Note that entries lacking a "collections" array are assumed to include all collections
CORPORA = {
    "fora-public": {
        "name": "Pubic Fora conversations",
        "conversations": get_public_conversations(),
        "examples": [
            "What are some themes related to gentrification?",
            "What are people's hopes and fears about artificial intelligence?",
            "Find the comment where a child asks her parent if the election of Trump meant that they would become slaves again",
            "Which kinds of jobs do people have where they work with animals?",
            "What can communities do to combat chronic absenteeism in schools?",
            "Teachers need to stop having to pay for their own classroom supplies",
        ],
    },
}

# Where to store the various search index files
INDEX_DIR = "indexes"


def display_speaker_name(name):
    return name.replace("Joe Rogan Experience", "JRE")


def playback_link(conv_id, start_time):
    if isinstance(conv_id, str):
        return youtube_link(conv_id, start_time)
    else:
        return "https://app.fora.io/conversation/%s?t=%f" % (conv_id, start_time)


def youtube_link(conv_id, start_time):
    return "https://www.youtube.com/watch?v={vid_id}&t={start_time}".format(
        vid_id=conv_id, start_time=start_time
    )


def load_all_data(use_local_embeddings=True):
    data = {}
    for c in CORPORA.keys():
        data[c] = load_data(use_local_embeddings, corpus=c)
    return data


def load_data(use_local_embeddings=True, corpus="all"):
    faiss_index = vectorize.get_faiss_index(
        use_local_embeddings=use_local_embeddings, corpus=corpus
    )
    docs = {}
    convs = {}
    speaker_intros = (
        {}
    )  # (conv_id, speaker_name) -> (snippet_index, first_snippet_spoken)
    with open(os.path.join(INDEX_DIR, corpus, "snippets.jsonl")) as fs:
        for line in fs:
            x = json.loads(line)
            docs[x["snippet_index"]] = x

            # Keep track of first turn for each speaker in each conversation
            conversation_id = x["conversation_id"]
            speaker_name = x["speaker_name"]
            index_in_conversation = x["index_in_conversation"]
            content = x["content"]
            intro_content = speaker_intros.get((conversation_id, speaker_name), [])
            if len(intro_content) < 5:
                intro_content.append(content)
                speaker_intros[(conversation_id, speaker_name)] = intro_content

    with open(os.path.join(INDEX_DIR, corpus, "conversations.jsonl")) as fs:
        for line in fs:
            x = json.loads(line)
            convs[x["id"]] = x
    return {
        "faiss_index": faiss_index,
        "conversations": convs,
        "docs": docs,
        "speaker_intros": speaker_intros,
    }


def run_query(user_input, data, encode, input_scope, corpus):
    embedding = encode([user_input])
    query_vec = np.zeros((1, embedding.shape[1])).astype("float32")
    query_vec[0] = embedding
    if input_scope.startswith("top_"):
        num_results = int(input_scope.split("_")[1])
    else:
        num_results = 100
    distances, items = data["faiss_index"].search(query_vec, num_results)
    results = []
    for i, res_idx in enumerate(items[0]):
        if res_idx < 0:
            continue
        res = dict(data["docs"][res_idx])
        res["res_idx"] = i + 1
        if "with_bio" in input_scope:
            res["speaker_intro"] = " ".join(
                data["speaker_intros"].get(
                    (res["conversation_id"], res["speaker_name"]), []
                )
            )
        if "with_context" in input_scope:
            res["prev_speaker_name"] = data["docs"][res_idx - 1]["speaker_name"]
            res["prev_content"] = data["docs"][res_idx - 1]["content"]
            res["next_speaker_name"] = data["docs"][res_idx + 1]["speaker_name"]
            res["next_content"] = data["docs"][res_idx + 1]["content"]

        if "context_window" in CORPORA[corpus]:
            match_sentence = data["docs"][res_idx]["content"]
            res["content"] = ""
            for j in range(res_idx - CORPORA[corpus]["context_window"], res_idx):
                res["content"] += " " + data["docs"][j]["content"]
            res["content"] += " **" + match_sentence + "**"
            for j in range(
                res_idx + 1, res_idx + CORPORA[corpus]["context_window"] + 1
            ):
                res["content"] += " " + data["docs"][j]["content"]

        results.append(res)

    return results


def run_rag_query(user_input, results, model, input_scope):
    if "with_bio" in input_scope:
        examples = [
            '- [%s] "%s" (from %s, whose first remarks were: "%s")'
            % (r["res_idx"], r["content"], r["speaker_name"], r["speaker_intro"])
            for r in results
        ]
    elif "with_context" in input_scope:
        examples = [
            '- [%s] %s: "%s"\n%s: "%s"\n%s: "%s"'
            % (
                r["res_idx"],
                r["prev_speaker_name"],
                r["prev_content"],
                r["speaker_name"],
                r["content"],
                r["next_speaker_name"],
                r["next_content"],
            )
            for r in results
        ]
    else:
        examples = ['- [%s] "%s"' % (r["res_idx"], r["content"]) for r in results]
    example_str = "\n".join(examples)
    prompt = (
        "Below are remarks from conversations, with an ID shown in brackets:\n\n###\n\n"
        + example_str
    )
    prompt += (
        "\n\n###\n\n"
        + "Please be sure to cite by ID in brackets, and answer the following: "
        + user_input
        + "\n\n"
    )
    prompt = gpt_lib.adjust_prompt(prompt, max_words=5000)
    summary = gpt_lib.run_gpt_query(prompt, model=model)
    return summary


def comma_replacer(match):
    # Extract match without surrounding parentheses and replace ', ' with ' ' for correct spacing
    return " ".join(match.group(0).translate({ord(c): None for c in "(),"}).split())


def analyze_citations(md, results, corpus=None):
    """Replace comment IDs in the summary with links to the relevant highlights."""
    citation_counts = defaultdict(lambda: 0)  # result_idx -> count

    # In parenthesizd lists of citations, remove commas and parens
    md = re.sub(r"\(\[[^\]]+\](, \[[^\]]+\])*\)", comma_replacer, md)

    for res in results:
        # TODO:  figure out the right regex here instead of using replace
        for key in [
            "[%s]" % (res["res_idx"]),
            "[%s," % (res["res_idx"]),
            " %s," % (res["res_idx"]),
            " %s]" % (res["res_idx"]),
        ]:
            if corpus == "joerogan":
                replacement_text = "[[:headphones: %s]](%s)" % (
                    display_speaker_name(res["speaker_name"]),
                    playback_link(res["conversation_id"], res["audio_start_offset"]),
                )
            else:
                replacement_text = " <sup>[%s (#%s)](#cite-%s)</sup> " % (
                    display_speaker_name(res["speaker_name"]),
                    res["res_idx"],
                    res["res_idx"],
                )
            citation_counts[res["res_idx"]] += md.count(key)
            md = md.replace(key, replacement_text)
    return md, citation_counts


def convert_results_to_csv(results):
    return pd.DataFrame(results).to_csv(index=False).encode("utf-8")
