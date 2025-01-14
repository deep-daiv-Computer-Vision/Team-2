# ======================= [built-in modules] =======================
import os
import time

# ====================== [third-party modules] =====================
import yaml
from box import Box
import numpy as np
from datasets import load_dataset
from bert_score import score as bert_score
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt

# ======================= [custom modules] =========================
from utils.eval_similarity import *
from utils.utils import *
from utils.segment_embedding import *
from utils.concat_functions import *
from utils.summarizer import *


# ========================= [Load config] ===========================
with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config)

print('Experiment name:', config.experiment_name)
print('===============================================')

# ========================== [Load data] ============================
print("Loading data... ", end="", flush=True)
if config.data.source == 'opensource':
    datasets = load_dataset(config.data.opensource)
    indices = np.load(f'data/gov_indices{config.data.index_set}.npy')
    datasets = datasets['train'].select(indices)['report']

elif config.data.source == 'youtube':
    datasets = load_dataset(config.data.youtube)
    indices = np.load(f'data/ytb_indices{config.data.index_set}.npy')
    datasets = datasets['train'].select(indices)['content']
print("Done")
print('===============================================')

save_dir_path = os.path.join('experiments', f'{config.experiment_name}')
if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)

# ========================== [Run experiments] ==========================
def run_experiment(datasets, config):
    max_score = 0
    best_summary = ""

    evaluation_results = []
    for di, text in enumerate(datasets):
        print(f" ----------------- [{di+1}/{len(datasets)}] ----------------- ")
        init_s = time.time()

        # ========================== [Segmentation] ========================
        print("Segmentating... ", end="", flush=True)
        s = time.time()
        segments = segmentate_sentence(text, **config.segment.args)
        e = time.time()
        print("Done", f"{e-s:.2f} sec")

        # ========================== [Clustering] ==========================
        print("Clustering...   ", end="", flush=True)
        s = time.time()
        concat_indices = globals()[config.concat.method](segments, **config.concat.args)
        e = time.time()
        print("Done", f"{e-s:.2f} sec")

        max_group_size = max([len(group) for group in concat_indices])
        avg_group_size = np.mean([len(group) for group in concat_indices])
        print(f"Num. of Cluster: {len(concat_indices)}, Max group size: {max_group_size}, Avg. group size: {avg_group_size:.2f}")

        # ========================== [Ready to summarize] ==================
        batch_clusters = [ #주제별로 문장들이 합쳐져서 있음
            " ".join([segments[gi] for gi in group]) for group in concat_indices
        ]

        # ========================== [Summarize] ===========================
        print("Summarizing...  ", end="", flush=True)
        s = time.time()
        if config.mini_batch.size > 0:
            mini_batch_size = (len(batch_clusters)
                               if len(batch_clusters) < config.mini_batch.size else
                               config.mini_batch.size)

            batch_summaries = []
            batch_token_importances = []
            for i in range(0, len(batch_clusters), mini_batch_size):
                mini_batch_summaries, mini_batch_token_importance = summarizer(batch_clusters[i:i+mini_batch_size], **config.summary.args)
                batch_summaries.append(mini_batch_summaries)
                batch_token_importances.append(mini_batch_token_importance)
            batch_summaries = " ".join(batch_summaries)
            # token_importance를 합치거나 평균을 내는 로직이 필요할 수 있습니다.
            token_importance = np.mean(batch_token_importances, axis=0)
        else:
            batch_summaries, token_importance = summarizer(batch_clusters, **config.summary.args)
        e = time.time()
        print("Done", f"{e-s:.2f} sec")

        # ========================== [Evaluate] ============================
        print("Evaluating...   ", end="", flush=True)
        s = time.time()
        
        rouge1, rouge2, rougeL = calculate_rouge_scores(text, batch_summaries)
        b_score = calculate_bert_score(text, batch_summaries)

        # scale score * 100
        rouge1, rouge2, rougeL = rouge1*100, rouge2*100, rougeL*100
        b_score = b_score * 100

        # 요약 결과와 성능 값을 저장합니다.
        evaluation_results.append({
            'summary': batch_summaries,
            'rouge1': rouge1,
            'rouge2': rouge2,
            'rougeL': rougeL,
            'bert_score': b_score,
            'token_importance': token_importance.tolist(),
            # 'visualization': visualization_path  # 그래프 시각화 경로 추가 필요
        })

    # 모든 결과를 반환합니다.
    return evaluation_results

# ========================== [Post-process] ========================
if b_score > max_score: # score는 대소비교 가능한 1가지 방식을 이용
    max_score = b_score
    best_summary = batch_summaries
    best_index = di
    # 원본 텍스트의 index는 indices[di]로 찾을 수 있음

evaluation_results.append({
    'rouge1': rouge1,
    'rouge2': rouge2,
    'rougeL': rougeL,
    'bert_score': b_score
})
print(f"Total: {time.time()-init_s:.2f} sec")

# append summary and scores to text file (cummulative)
# if there is no file, create one
if config.save_summaries:
    with open(f'experiments/{config.experiment_name}/summaries.txt', 'a') as f:
        f.write(f"==================== [{di+1}/{len(datasets)}] ====================\n")
        # f.write(f"Original text:\n{text}\n\n")
        f.write(f"Summary:\n{batch_summaries}\n\n")
        f.write(f"ROUGE-1: {rouge1:.2f}, ROUGE-2: {rouge2:.2f}, ROUGE-L: {rougeL:.2f}\n")
        f.write(f"BERTScore: {b_score:.2f}\n\n")
        f.write("==============================================\n")


