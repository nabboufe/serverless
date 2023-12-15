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
        'batch_size': 1,
        'params': False,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    response = requests.post(URI, json=dict(input=request), headers=headers)
    data = response.json()
    task_id = data.get('id')
    status = f"https://api.runpod.ai/v2/fimwy543smiqak/status/{task_id}"

    while True :
        response = requests.post(status, json=dict(input=request), headers=headers)
        print(f"response: {response}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"data: {data}")
        
        if data["status"] != "IN_QUEUE" and data["status"] != "IN_PROGRESS" :
            print(data["output"])
            break
        
        sleep(3)

def cancel_task(task_id):
    url = f"https://api.runpod.ai/v2/{endpoint_id}/cancel/{task_id}"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    response = requests.get(url, headers=headers)
    print(response)
    return response


if __name__ == '__main__':

    prompt = """"""

    import time
    start = time.time()
    print(run(prompt))
    print("Time taken: ", time.time() - start, " seconds")