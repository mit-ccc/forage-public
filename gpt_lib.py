#!/usr/bin/env python3

"""
Helper script for OpenAI API.  Be sure OPENAI_API_KEY is set in the environment.
"""

import os
import sys
import logging
import random

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    before_log,
)
from openai import OpenAI

client = OpenAI()

# Set to a model ID supported by the OpenAI ChatCompletions API
# See:  https://platform.openai.com/docs/models
_MODEL = "o1"

if "GPT_LIB_LOGLEVEL" in os.environ.keys():
    loglevel = getattr(logging, os.environ["GPT_LIB_LOGLEVEL"])  # 'INFO', etc
else:
    loglevel = logging.INFO

logging.basicConfig(stream=sys.stderr, level=loglevel)
logger = logging.getLogger(__name__)


def run_gpt_query(prompt, model=_MODEL):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2000,
    )
    result = response.choices[0].message.content
    return result


def adjust_prompt(prompt, max_words=20000):
    """If the prompt exceeds max_words, remove randomly selected lines until it doesn't.
    Do not delete the first or the last few lines to avoid deleting instructional text.
    """
    attempts = 0
    while len(prompt.split()) > max_words and attempts < 1000:
        prompt_lines = prompt.split("\n")
        prompt_lines = [x[:200] for x in prompt_lines]
        if len(prompt_lines) < 3:
            break
        remove_line = random.randint(1, len(prompt_lines) - 3)
        if not prompt_lines[remove_line].endswith(":"):
            del prompt_lines[remove_line]
            prompt = "\n".join(prompt_lines)
        attempts += 1
    return prompt
