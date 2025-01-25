from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

def download_model(model_name):
    local_path = f"./models_installed/{model_name}"
    
    if not os.path.exists(local_path):
        os.makedirs(local_path)
        
    # 모델과 토크나이저 다운로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # 로컬에 저장
    tokenizer.save_pretrained(local_path)
    model.save_pretrained(local_path)
    print(f"Downloaded {model_name}")

# 사용할 모델들 미리 다운로드
models = [
    "facebook/bart-large-cnn"
]

for model_name in models:
    download_model(model_name)
