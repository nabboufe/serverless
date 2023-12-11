from huggingface_hub import snapshot_download, HfFolder

model_directory = snapshot_download("affecto/Vigogne70b-last_fan", revision="main", local_dir="/home/ubuntu/serverless/models/Vigogne/base")
lora_directory = snapshot_download("TheBloke/vigogne-2-70B-chat-GPTQ", revision="main", local_dir="/home/ubuntu/serverless/models/Vigogne/LORA")
classifier = snapshot_download("mtheo/camembert-base-xnli", revision="main", local_dir="/home/ubuntu/serverless/models/classifier")

print(lora_directory, model_directory, classifier)
