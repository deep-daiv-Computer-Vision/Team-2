from transformers import BartTokenizer, BartForConditionalGeneration
import torch

def calculate_token_importance_gradient(input_text, model_name="facebook/bart-large-cnn", max_length=1024):
    """
    Calculate token importance using gradients of the input embeddings.

    Args:
        input_text (str): The input text to analyze.
        model_name (str): Pretrained model name (default: facebook/bart-large-cnn).
        max_length (int): Maximum length of the input text (default: 1024).

    Returns:
        dict: A dictionary mapping tokens to their importance scores.
    """
    # 모델과 토크나이저 로드
    model = BartForConditionalGeneration.from_pretrained(model_name, output_attentions=True)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    # 텍스트를 토큰화
    inputs = tokenizer(input_text, return_tensors='pt', max_length=max_length, truncation=True)

    # 입력 임베딩 활성화
    inputs_embeds = model.get_input_embeddings()(inputs["input_ids"])
    inputs_embeds = inputs_embeds.clone().detach()  # Leaf 텐서로 변환
    inputs_embeds.requires_grad = True  # 기울기 계산 활성화

    # Forward pass
    outputs = model(inputs_embeds=inputs_embeds, decoder_input_ids=inputs["input_ids"])

    # 모델 출력에서 첫 번째 토큰의 확률 합 선택
    output_logits = outputs.logits[:, 0, :]  # 첫 번째 토큰의 로짓
    target_class = output_logits.max(dim=1)[1]  # 확률이 가장 높은 클래스 선택
    target_score = output_logits[:, target_class]  # 선택된 클래스의 점수

    # Backward pass (기울기 계산)
    target_score.sum().backward()

    # 입력 임베딩의 기울기 추출
    gradients = inputs_embeds.grad  # 입력 임베딩에 대한 기울기

    # 기울기 중요도 계산 (기울기의 절댓값 평균)
    gradients_mean = gradients.abs().mean(dim=2)  # Embedding 차원 평균

    # 입력 토큰과 중요도 매핑
    input_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
    token_importance = {
        token: score.item() 
        for token, score in zip(input_tokens, gradients_mean.squeeze())
    }

    return token_importance
