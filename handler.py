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

import os
import logging
from typing import Generator, Union
import runpod
from huggingface_hub import snapshot_download, HfFolder

from copy import copy

logging.info("v5 work? test")

model_directory = "/runpod-volume/hub/Vigogne/base/"
lora_directory = "/runpod-volume/hub/Vigogne/LORA/"

config = ExLlamaV2Config()
config.model_dir = model_directory
config.prepare()

model = ExLlamaV2(config)
model.load()

tokenizer = ExLlamaV2Tokenizer(config)
cache = ExLlamaV2Cache(model)

lora = ExLlamaV2Lora.from_directory(model, lora_directory)

simple_generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

def inference(event) -> Union[str, Generator[str, None, None]]:

    prompt = event["input"]["prompt"]
    max_new_token = event["input"]["max_new_tokens"]

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = event["input"]["temperature"]
    settings.top_k = event["input"]["top_k"]
    settings.top_p = event["input"]["top_p"]
    settings.token_repetition_penalty = event["input"]["repetition_penalty"]

    print(f"generator: {simple_generator}. lora: {lora}")
    print(f"prompt: {prompt}. settings: {settings}. max_new_token: {max_new_token}")

    output = simple_generator.generate_simple(prompt, settings, max_new_token, loras = lora)

    return output[len(prompt):]

runpod.serverless.start({"handler": inference})
