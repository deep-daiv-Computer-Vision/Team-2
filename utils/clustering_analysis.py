import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from .segment_embedding import encode_segments
from matplotlib.colors import ListedColormap
import umap

def analyze_text_clusters(segments: list, index: list, eps: float=0.3, min_samples: int=2):
    """
    원본 텍스트와 요약 텍스트를 받아 클러스터링 분석 및 시각화를 수행하는 함수
    
    Args:
        original_text (list): 원본 텍스트
        summary_text (str): 요약 텍스트
        eps (float): DBSCAN의 epsilon 파라미터
        min_samples (int): DBSCAN의 최소 샘플 수 파라미터
    
    Returns:
        tuple: (labels, embeddings_2d, original_sentences, summary_sentences)
    """
    
    theme_sentences = []
    for idx, sentence_idx in enumerate(index):
        tmp_sentence = []

        for st_idx in sentence_idx:
            tmp_sentence.append(segments[st_idx])
        
        theme_sentences.append(tmp_sentence)
    
    return analyze_sentences(theme_sentences, eps, min_samples)

def analyze_sentences(original_sentences: list, eps: float=0.3, min_samples: int=2, visualize_pth: str='umap.png'):
    """
    문장들을 임베딩하고 클러스터링하여 시각화하는 함수
    
    Args:
        original_sentences (list): 원본 문장 리스트
        summary_sentences (list): 요약 문장 리스트
        eps (float): DBSCAN의 epsilon 파라미터
        min_samples (int): DBSCAN의 최소 샘플 수 파라미터
    
    Returns:
        tuple: (labels, embeddings_2d)
    """
    all_sentences = original_sentences
    all_embeddings = []

    for sentence in all_sentences:
        embeddings = encode_segments(sentence)
        all_embeddings.append(embeddings)

    flattened_embeddings = np.vstack(all_embeddings)  # 모든 theme의 embedding을 하나로 합침
    theme_labels = []  # 각 theme에 해당하는 레이블 생성
    for i, embeddings in enumerate(all_embeddings):
        theme_labels.extend([i] * len(embeddings))  # theme ID를 레이블로 추가

    theme_labels = np.array(theme_labels)

    reducer = umap.UMAP(metric='cosine', n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    reduced_embeddings = reducer.fit_transform(flattened_embeddings)

    tab10 = plt.cm.get_cmap('tab10', 10)
    colors = tab10.colors[:5]
    cmap = ListedColormap(colors)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=theme_labels,
        cmap=cmap,
        s=10
    )
    plt.colorbar(scatter, ticks=range(5), label="Theme ID")
    plt.title("UMAP Visualization of Embeddings by Theme")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.savefig(visualize_pth)

    return visualize_pth

## 시각화 모듈 사용 방법
# analyze_text_clusters(segments, concat_indices)
