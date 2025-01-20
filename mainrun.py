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
from utils.clustering_analysis import *

def exe_by_sentences(text: str):
    # ========================= [Load config] ===========================
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = Box(config)


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
    batch_clusters = [
        " ".join([segments[gi] for gi in group]) for group in concat_indices
    ]
    
    visualize_pth = analyze_text_clusters(segments, concat_indices)

    # ========================== [Summarize] ===========================
    print("Summarizing...  ", end="", flush=True)
    s = time.time()
    if config.mini_batch.size > 0:
        mini_batch_size = (len(batch_clusters)
                           if len(batch_clusters) < config.mini_batch.size else
                           config.mini_batch.size)

        batch_summaries = []
        batch_importances = []
        for i in range(0, len(batch_clusters), mini_batch_size):
            mini_batch_summaries, mini_batch_importances = summarizer(batch_clusters[i:i+mini_batch_size], cal_grad=True, **config.summary.args)
            batch_summaries.append(mini_batch_summaries)
            batch_importances.append(mini_batch_importances)
        total_summaries = " ".join(batch_summaries)
    else:
        batch_summaries = summarizer(batch_clusters, **config.summary.args)
    e = time.time()
    print("Done", f"{e-s:.2f} sec")

    # ========================== [Evaluate] ============================
    print("Evaluating...   ", end="", flush=True)
    s = time.time()
    
    rouge1, rouge2, rougeL = calculate_rouge_scores(text, total_summaries)
    s_score = calculate_sementic_similarity(text, total_summaries)

    # scale score * 100
    rouge1, rouge2, rougeL = rouge1*100, rouge2*100, rougeL*100
    s_score = s_score * 100

    e = time.time()
    print("Done", f"{e-s:.2f} sec")
    
    print(f"=> ROUGE-1: {rouge1:.2f}, ROUGE-2: {rouge2:.2f}, ROUGE-L: {rougeL:.2f}")
    print(f"=> BERTScore: {s_score:.2f}")

    # ========================== [Post-process] ========================
    evaluation_results= {
        'rouge1': rouge1,
        'rouge2': rouge2,
        'rougeL': rougeL,
        'bert_score': s_score
    }

    return batch_summaries, batch_importances, evaluation_results, visualize_pth

def resummarize_with_sentece(full_text: str, target_text: str):
    # ========================= [Load config] ===========================
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = Box(config)

    # ========================== [Segmentation] ========================
    print("Segmentating... ", end="", flush=True)
    s = time.time()
    segments = segmentate_sentence(full_text, **config.segment.args)
    e = time.time()
    print("Done", f"{e-s:.2f} sec")

    # ========================== [Filtering] ==========================
    print("Filtering...   ", end="", flush=True)

    filtered_text = []
    for segment in segments:
        if calculate_semantic_similarity(segment, target_text) > 0.8:
            filtered_text.append(segment)
    
    filtered_text = " ".append(filtered_text)

    # ========================== [Summarize] ===========================
    print("Summarizing...  ", end="", flush=True)
    batch_summaries = summarizer(filtered_text, cal_grad=False, **config.summary.args)

    return batch_summaries

# # ========================= [Load config] ===========================
# with open("config.yaml", "r") as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)
#     config = Box(config)

# print('Experiment name:', config.experiment_name)
# print('===============================================')

# # ========================== [Run experiments] ==========================
# def summarize_and_visualization(text, config):
#     max_score = 0
#     best_summary = ""
#     best_index = 0

#     evaluation_results = []

#     init_s = time.time()

#     # ========================== [Segmentation] ========================
#     print("Segmentating... ", end="", flush=True)
#     s = time.time()
#     segments = segmentate_sentence(text, **config.segment.args)
#     e = time.time()
#     print("Done", f"{e-s:.2f} sec")

#     # ========================== [Clustering] ==========================
#     print("Clustering...   ", end="", flush=True)
#     s = time.time()
#     concat_indices = globals()[config.concat.method](segments, **config.concat.args)
#     e = time.time()
#     print("Done", f"{e-s:.2f} sec")

#     max_group_size = max([len(group) for group in concat_indices])
#     avg_group_size = np.mean([len(group) for group in concat_indices])
#     print(f"Num. of Cluster: {len(concat_indices)}, Max group size: {max_group_size}, Avg. group size: {avg_group_size:.2f}")

#     # 여기서 클러스터링한 걸 가지고 아래 불러온 함수에서 시각화를 합시다~~~

#     # ========================== [Ready to summarize] ==================
#     batch_clusters = [ #주제별로 문장들이 합쳐져서 있음
#         " ".join([segments[gi] for gi in group]) for group in concat_indices
#     ]

#     # ========================== [Summarize] ===========================
#     print("Summarizing...  ", end="", flush=True)
#     s = time.time()
#     if config.mini_batch.size > 0:
#         mini_batch_size = (len(batch_clusters)
#                            if len(batch_clusters) < config.mini_batch.size else
#                            config.mini_batch.size)

#         batch_summaries = []
#         batch_token_importances = []
#         for i in range(0, len(batch_clusters), mini_batch_size):
#             mini_batch_summaries, mini_batch_token_importance = summarizer(batch_clusters[i:i+mini_batch_size], **config.summary.args)
#             batch_summaries.append(mini_batch_summaries)
#             batch_token_importances.append(mini_batch_token_importance)
#         batch_summaries = " ".join(batch_summaries)
#         # token_importance를 합치거나 평균을 내는 로직이 필요할 수 있습니다.
#         token_importance = np.mean(batch_token_importances, axis=0)
#     else:
#         batch_summaries, token_importance = summarizer(batch_clusters, **config.summary.args)
#     e = time.time()
#     print("Done", f"{e-s:.2f} sec")

#     # ========================== [Evaluate] ============================
#     print("Evaluating...   ", end="", flush=True)
#     s = time.time()
    
#     rouge1, rouge2, rougeL = calculate_rouge_scores(text, batch_summaries)
#     b_score = calculate_bert_score(text, batch_summaries)

#     # scale score * 100
#     rouge1, rouge2, rougeL = rouge1*100, rouge2*100, rougeL*100
#     b_score = b_score * 100
    
#     # ========================== [Post-process] ========================
#     if b_score > max_score: # score는 대소비교 가능한 1가지 방식을 이용
#         max_score = b_score
#         best_summary = batch_summaries
#         best_index = 0

#     evaluation_results.append({
#         'summary': batch_summaries,
#         'rouge1': rouge1,
#         'rouge2': rouge2,
#         'rougeL': rougeL,
#         'bert_score': b_score,
#         'token_importance': token_importance.tolist(),
#         # 'visualization': visualization_path  # 그래프 시각화 경로 추가 필요
#     })

#     # 모든 결과를 반환합니다.
#     return evaluation_results

# def brushing_and_resummarize(datasets, config, selected_text):
#     """
#     사용자가 선택한 텍스트와 전체 텍스트의 유사도를 기반으로 요약을 생성.

#     Args:
#     - datasets (list of str): 전체 텍스트 리스트.
#     - config (Box): 설정 객체.  -> 솔직히 이거 왜 필요한가 싶음
#     - selected_text (str): 사용자가 선택한 텍스트.

#     Returns:
#     - list of dict: 각 텍스트에 대한 요약 및 평가 결과.
#     """
#     results = []
#     # 전체 텍스트를 문장 단위로 분할
#     sentences = datasets.split('. ')
    
#     # 각 문장과 선택된 텍스트의 유사도 계산
#     similarities = [calculate_semantic_similarity(sentence, selected_text) for sentence in sentences]
    
#     # 유사도가 높은 순으로 정렬하여 상위 n개의 문장 선택
#     n = 3  # 요약에 포함할 문장 수
#     top_sentences = [sentences[i] for i in np.argsort(similarities)[-n:]]
    
#     # 선택된 문장들을 결합하여 요약 생성
#     summary = '. '.join(top_sentences)
    

#     result = {
#         'summary': summary,
#     }

#     return result
