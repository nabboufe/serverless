import os
import requests
from time import sleep
import logging
import argparse
import sys
import json

API_KEY = "POJKJKXH5F213A8HO7A9ARD7T6XQIDPUM1258ZY9"

endpoint_id = "fimwy543smiqak"
URI = f"https://api.runpod.ai/v2/{endpoint_id}/run"

def run(prompt):
    request = {
        'prompt': prompt,
        'max_new_tokens': 128,
        'temperature': 0.7,
        'top_k': 30,
        'top_p': 0.95,
        'repetition_penalty': 1.1,
        'batch_size': 8,
        'params': False,
    }

    response = requests.post(URI, json=dict(input=request), headers = {
        "Authorization": f"Bearer {API_KEY}"
    })

    if response.status_code == 200:
        data = response.json()
        task_id = data.get('id')
        return stream_output(task_id)


def stream_output(task_id, stream=False):
    url = f"https://api.runpod.ai/v2/{endpoint_id}/stream/{task_id}"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    previous_output = ''

    try:
        while True:
            response = requests.get(url, headers=headers)
            print(f"response: {response}")
            if response.status_code == 200:
                data = response.json()
                print(f"data: {data}")
                if data.get('status') == 'COMPLETED':
                    if not stream:
                        return previous_output
                    break
            elif response.status_code >= 400:
                print(response)
            # Sleep for 0.1 seconds between each request
            sleep(0.1 if stream else 1)
    except Exception as e:
        print(e)
        cancel_task(task_id)
    

def cancel_task(task_id):
    url = f"https://api.runpod.ai/v2/{endpoint_id}/cancel/{task_id}"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    response = requests.get(url, headers=headers)
    print(response)
    return response


if __name__ == '__main__':

    prompt = """Salut ca va ?"""

    import time
    start = time.time()
    print(run(prompt))
    print("Time taken: ", time.time() - start, " seconds")