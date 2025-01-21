import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# pip install rouge_score bert_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from .segment_embedding import *
from .utils import cosine_similarity

def calculate_rouge_scores(original_text, summary):
    """
    ROUGE-1, ROUGE-2, ROUGE-L 점수를 계산.
    
    Args:
    - original_text (str): 원본 텍스트.
    - summary (str): 요약 텍스트.
    
    Returns:
    -  'rouge1': float, 'rouge2': float, 'rougeL': float.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(original_text, summary)
    return (
        scores['rouge1'].fmeasure,
        scores['rouge2'].fmeasure,
        scores['rougeL'].fmeasure
    )

def calculate_bert_score(original_text, summary, model="bert-base-uncased"):
    """
    BERTScore를 계산.
    
    Args:
    - original_text (str): 원본 텍스트.
    - summary (str): 요약 텍스트.
    - model (str): 사용할 BERT 모델의 이름 (default: "bert-base-uncased").
    
    Returns:
    - float: BERTScore.
    """
    P, R, F = bert_score([summary], [original_text], model_type=model, lang="en")
    return F.mean().item()


def calculate_rouge_matrices(texts):
    """
    주어진 텍스트 리스트에 대해 ROUGE-1, ROUGE-2, ROUGE-L 매트릭스를 계산.

    Args:
    - texts (list of str): 비교할 텍스트 리스트.

    Returns:
    - tuple of np.ndarray: (rouge1_matrix, rouge2_matrix, rougeL_matrix)
        - rouge1_matrix: ROUGE-1 F-measure 매트릭스 (n x n).
        - rouge2_matrix: ROUGE-2 F-measure 매트릭스 (n x n).
        - rougeL_matrix: ROUGE-L F-measure 매트릭스 (n x n).
    """
    n = len(texts)
    rouge1_matrix = np.zeros((n, n))
    rouge2_matrix = np.zeros((n, n))
    rougeL_matrix = np.zeros((n, n))
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                scores = scorer.score(texts[i], texts[j])
                rouge1_matrix[i][j] = scores['rouge1'].fmeasure
                rouge2_matrix[i][j] = scores['rouge2'].fmeasure
                rougeL_matrix[i][j] = scores['rougeL'].fmeasure
            else:
                # Self-comparison should always be 1.0
                rouge1_matrix[i][j] = 1.0
                rouge2_matrix[i][j] = 1.0
                rougeL_matrix[i][j] = 1.0
                
    return rouge1_matrix, rouge2_matrix, rougeL_matrix

def calculate_bert_matrix(texts, model="bert-base-uncased"):
    """
    주어진 텍스트 리스트에 대해 BERTScore 매트릭스를 계산.

    Args:
    - texts (list of str): 비교할 텍스트 리스트.
    - model (str): 사용할 BERT 모델의 이름 (default: "bert-base-uncased").

    Returns:
    - np.ndarray: BERTScore 매트릭스 (n x n).
    """
    n = len(texts)
    bert_matrix = np.zeros((n, n))
    for i in range(n):
        refs = [texts[j] for j in range(n)]
        _, _, scores = bert_score([texts[i]] * n, refs, model_type=model, lang="en")
        bert_matrix[i, :] = scores.numpy()
        
    return bert_matrix


# dlrjek
def calculate_semantic_similarity(original_text, summary, sent2vec=True):
    """
    Sent2vec의 유사도를 이용한 sementic_similarity 계산.
    
    Args:
    - original_text (str): 원본 텍스트.
    - summary (str): 요약 텍스트.
    - model (str): 사용할 BERT 모델의 이름 (default: "bert-base-uncased").
    
    Returns:
    - float: sementic_similarity.
    """
    embeddings = encode_segments([summary, original_text])
    sementic_similarity = cosine_similarity(embeddings[0], embeddings[1])

    return sementic_similarity

def plot_heatmap(matrix, title):
    """
    주어진 매트릭스를 히트맵 형태로 시각화.

    Args:
    - matrix (np.ndarray): 시각화할 n x n 매트릭스.
    - title (str): 히트맵의 제목.

    Returns:
    - None: 히트맵을 출력.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.xlabel("Texts")
    plt.ylabel("Texts")
    plt.show()


# Example usage
if __name__ == '__main__':
    texts = [
        "The cat sat on the mat.",
        "The mat was sat on by the cat.",
        "A dog barked at the cat on the mat.",
        "Cats love sitting on mats."
    ]

    rouge1_matrix, rouge2_matrix, rougeL_matrix = calculate_rouge_matrices(texts)

    bert_matrix = calculate_bert_matrix(texts)

    plot_heatmap(rouge1_matrix, "ROUGE-1 Score Matrix")
    plot_heatmap(rouge2_matrix, "ROUGE-2 Score Matrix")
    plot_heatmap(rougeL_matrix, "ROUGE-L Score Matrix")

    plot_heatmap(bert_matrix, "BERTScore Matrix")
