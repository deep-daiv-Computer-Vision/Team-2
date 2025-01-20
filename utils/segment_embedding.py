from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def segmentate_sentence(full_text: str, n_word: int, n_overlap: int=0, fix_size: bool=False, by_sentences: bool=False) -> List[str]:
    """
    전체 텍스트를 n_word 단어 개수로 나누어 리스트로 반환

    Args:
    - full_text: 전체 텍스트
    - n_word: 나누어질 단어 개수
    - n_overlap: 나누어진 문장 간 겹칠 단어 개수
    - fix_size: 마지막 문장이 n_word보다 작을 때, 마지막 문장을 n_word로 맞추기

    Returns:
    - List[str]: 나누어진 문장 리스트
    """
    assert n_word > n_overlap, "n_word must be greater than n_overlap"

    if !by_sentences:
        words = full_text.split()
        # assert n_word <= len(words), "n_word must be less than the number of words in full_text"
    
        result = []
        for i in range(0, len(words), n_word-n_overlap):
            result.append(" ".join(words[i:i+n_word]))
    
            if i + n_word >= len(words):
                break
    
        if fix_size:
            result[-1] = " ".join(words[-n_word:])
    else:
        result = full_text.split('.')

    return result

def encode_segments(segments: List[str], model_name: str='sentence-transformers/all-MiniLM-L6-v2', normalize: int=2) -> np.ndarray:
    """
    segment list를 입력받아 embedding을 반환

    Args:
    - segments: segment list
    - model_name: model name

    Returns:
    - np.ndarray: embeddings
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    segment_tokens = tokenizer(segments, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**segment_tokens)

    embeddings = outputs[0]
    if 'attention_mask' in segment_tokens:
        # sentence-transformers dependent code, please ref https://huggingface.co/sentence-transformers
        mask = segment_tokens['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
        embeddings = torch.sum(embeddings * mask, dim=1) / torch.clamp(mask.sum(1), min=1e-9)
    else:
        embeddings = embeddings.mean(dim=1)

    if normalize:
        embeddings = F.normalize(embeddings, p=normalize, dim=1)

    return embeddings.numpy()

# Testing
if __name__ == "__main__":
    with open('sample_text.txt', 'r') as f:
        full_text = f.read()

    import time
    from utils import cosine_similarity

    n_word = 500
    n_overlap = 0

    s = time.time()
    segments = segmentate_sentence(full_text, n_word, n_overlap, False)
    print(f"segmentate_sentence: {time.time()-s:.2f}s")
    
    s = time.time()
    embeddings = encode_segments(segments)
    print(f"encode_segments: {time.time()-s:.2f}s")

    s = time.time()
    cos_sim = cosine_similarity(embeddings, embeddings)
    print(f"cosine_similarity: {time.time()-s:.2f}s")
    print(cos_sim, cos_sim.shape)
