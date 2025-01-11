import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import nltk
from nltk.tokenize import sent_tokenize

# NLTK 데이터 다운로드
nltk.download('punkt')
nltk.download('punkt_tab')

def analyze_text_clusters(original_text: str, summary_text: str, eps: float=0.3, min_samples: int=2):
    """
    원본 텍스트와 요약 텍스트를 받아 클러스터링 분석을 수행하는 함수
    
    Args:
        original_text (str): 원본 텍스트
        summary_text (str): 요약 텍스트
        eps (float): DBSCAN의 epsilon 파라미터
        min_samples (int): DBSCAN의 최소 샘플 수 파라미터
    
    Returns:
        tuple: (labels, embeddings_2d, original_sentences, summary_sentences)
    """
    original_sentences = sent_tokenize(original_text, language='english')
    summary_sentences = sent_tokenize(summary_text, language='english')
    
    return analyze_sentences(original_sentences, summary_sentences, eps, min_samples)

def analyze_sentences(original_sentences: list, summary_sentences: list, eps: float=0.3, min_samples: int=2):
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
    # 기존 analyze_sentences 함수의 내용
    all_sentences = original_sentences + summary_sentences
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(all_sentences)
    
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled_embeddings)
    labels = clustering.labels_
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_sentences)-1))
    embeddings_2d = tsne.fit_transform(scaled_embeddings)
    
    return labels, embeddings_2d, original_sentences, summary_sentences

def visualize_clusters(labels, embeddings_2d, original_sentences, summary_sentences):
    """
    클러스터링 결과를 시각화하는 함수
    
    Args:
        labels: 클러스터 레이블
        embeddings_2d: 2차원으로 축소된 임베딩
        original_sentences: 원본 문장 리스트
        summary_sentences: 요약 문장 리스트
    """
    plt.figure(figsize=(12, 8))
    
    n_original = len(original_sentences)
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask_orig = np.logical_and(labels == label, np.arange(len(labels)) < n_original)
        plt.scatter(embeddings_2d[mask_orig, 0], embeddings_2d[mask_orig, 1], 
                   c=[color], marker='o', s=100, label=f'Cluster {label}')
        
        mask_summ = np.logical_and(labels == label, np.arange(len(labels)) >= n_original)
        plt.scatter(embeddings_2d[mask_summ, 0], embeddings_2d[mask_summ, 1], 
                   c=[color], marker='^', s=150)
    
    plt.title('Sentence Clustering Visualization\n(△: 요약 문장, ○: 원본 문장)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.show()