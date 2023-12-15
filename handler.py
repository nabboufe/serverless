from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    ExLlamaV2Lora,
)

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)

import os, glob
import logging
from typing import Generator, Union
import runpod
from huggingface_hub import snapshot_download, HfFolder

from copy import copy

import re
import codecs

ESCAPE_SEQUENCE_RE = re.compile(r'''
    ( \\U........      # 8-digit hex escapes
    | \\u....          # 4-digit hex escapes
    | \\x..            # 2-digit hex escapes
    | \\N\{[^}]+\}     # Unicode characters by name
    | \\[\\'"abfnrtv]  # Single-character escapes
    )''', re.UNICODE | re.VERBOSE)

def decode_escapes(s):
    def decode_match(match):
        return codecs.decode(match.group(0), 'unicode-escape')

    return ESCAPE_SEQUENCE_RE.sub(decode_match, s)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

def load_model():
    global simple_generator, lora

    logging.info(f"generator: {not not simple_generator}, lora: {not not lora}")

    if not simple_generator or not lora :
        logging.info("simple generator")
        token_read = os.environ["HF_TOKEN"]
        HfFolder.save_token(token_read)

        model_directory = "/data/models/Vigogne/base"
        lora_directory = "/data/models/Vigogne/LORA"

        config = ExLlamaV2Config()
        config.model_dir = model_directory
        config.prepare()

        model = ExLlamaV2(config)
        model.load()

        tokenizer = ExLlamaV2Tokenizer(config)
        cache = ExLlamaV2Cache(model)

        lora = ExLlamaV2Lora.from_directory(model, lora_directory)

        simple_generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

    return simple_generator, lora

simple_generator = None
lora = None
default_settings = None

def inference(event) -> Union[str, Generator[str, None, None]]:

    logging.info(event)
    prompt = event["input"]["prompt"]
    max_new_token = event["input"]["max_new_tokens"]

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = event["input"]["temperature"]
    settings.top_k = event["input"]["top_k"]
    settings.top_p = event["input"]["top_p"]
    settings.token_repetition_penalty = event["input"]["repetition_penalty"]

    simple_generator, lora = load_model()

    logging.info(prompt, settings, max_new_token, lora)
    output = simple_generator.generate_simple(prompt, settings, max_new_token, loras = lora)

    return output[len(prompt):]

runpod.serverless.start({"handler": inference})
