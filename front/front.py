import streamlit as st
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    pipeline
)
import torch
from rouge_score import rouge_scorer
from bert_score import score

# TODO: 백엔드와 연결하는 작업 필요
# TODO: brushing POP-UP창 작업 필요

# 페이지 설정을 가장 먼저 호출
st.set_page_config(
    page_title="XAI Summarization",
    page_icon="🤖", layout="wide"
    )

def create_attention_html(text, attention_scores):
    """텍스트에 attention score를 적용하여 HTML로 변환"""
    html = ""
    for word, score in zip(text.split(), attention_scores):
        # score를 0-1 사이의 값으로 정규화
        intensity = int(score * 255)
        html += f'<span style="background-color: rgba(255, 0, 0, {score:.2f})">{word}</span> '
    return html

@st.cache_resource
def load_model(model_name):
    """모델과 토크나이저를 로드하고 캐시"""
    local_model_path = f"../models_installed/{model_name}"  # 로컬 모델 저장 경로
    # TODO: 서버에서 미리 모델 저장하고 부를건지 체크해야 함
    
    try:
        # 로컬에서 모델 로드 시도
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)
    except:
        # 로컬에 없으면 허깅페이스에서 다운로드 후 저장
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # 모델과 토크나이저 저장
        tokenizer.save_pretrained(local_model_path)
        model.save_pretrained(local_model_path)
    
    return model, tokenizer

# TODO:임의적으로 함수화하긴 했는데, codebase 단에 있는 함수로 교체해야 함
def get_summary_and_attention(text, model_name):
    """텍스트 요약 및 어텐션 스코어 계산"""
    model, tokenizer = load_model(model_name)
    
    # 토크나이징
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    
    # 요약 생성
    with torch.no_grad():
        # 먼저 요약문 생성
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )
        
        # attention weights 계산을 위한 forward pass
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_attentions=True
        )
    
    # 요약문 디코딩
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # 어텐션 스코어 계산 (인코더의 마지막 레이어 attention 사용)
    attention_weights = outputs.encoder_attentions[-1]  # 마지막 레이어의 어텐션
    attention_weights = torch.mean(attention_weights, dim=1)  # head 평균
    attention_scores = attention_weights[0].mean(dim=0)  # 시퀀스 평균
    
    # 입력 텍스트 길이에 맞게 자르기
    attention_scores = attention_scores[:len(text.split())].numpy()
    
    return summary, attention_scores

# TODO: 요약모델 계산 Codebase 단에 있는 함수로 교체
def calculate_rouge(summary, reference):
    """ROUGE 점수 계산"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    
    return {
        'rouge-1': {'f': scores['rouge1'].fmeasure},
        'rouge-2': {'f': scores['rouge2'].fmeasure},
        'rouge-l': {'f': scores['rougeL'].fmeasure}
    }

def calculate_bert_score(summary, reference):
    """BERTScore 계산"""
    P, R, F1 = score([summary], [reference], lang="en", verbose=False)
    return F1.mean().item()

def main():
    # 팝업 상태 관리를 위한 session state 초기화
    if 'popup_states' not in st.session_state:
        st.session_state.popup_states = {}

    # CSS 스타일 추가
    st.markdown("""
        <style>
        .text-section {
            position: relative;
            margin: 10px 0;
        }
        .summary-box {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-top: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("🔮시험 공부 벼락치기 시트 만들기")
    st.write("본 요약시스템은 영어 요약만 제공합니다.")
    # 사이드바 설정
    st.sidebar.title("강의 내용 요약하기")
    # 입력 방식 선택
    input_type = st.sidebar.segmented_control(
        label="입력 방식을 선택하세요",
        options=["파일 업로드", "직접 텍스트 입력"],
        default="파일 업로드"
    )
    
    # 입력 섹션
    st.sidebar.header("Input Section")
    text_input = None
    
    if input_type == "파일 업로드":
        uploaded_file = st.sidebar.file_uploader(
            "요약할 텍스트 파일(.txt)을 업로드 하세요.",
            type=['txt'],
            help="최대 200MB까지 업로드 가능합니다."
        )
        
        # TODO: 개발 어느정도 끝나면 이 부분 삭제
        if uploaded_file is not None:
            file_details = {
                "파일명": uploaded_file.name,
                "파일크기": f"{uploaded_file.size / 1024:.2f} KB",
                "파일타입": uploaded_file.type
            }
            st.sidebar.write("### 파일 정보")
            st.sidebar.json(file_details)
            # 파일 내용 읽기
            text_input = uploaded_file.getvalue().decode('utf-8')
            
    else:
        text_input = st.sidebar.text_area(
            "요약할 텍스트를 입력하세요",
            height=180,
            placeholder="여기에 요약할 텍스트를 입력하세요..."
        )

    # 구분선 추가
    st.sidebar.divider()
    
    # 요약 모델 선택 섹션
    st.sidebar.header("Summarization Model")
    
    # 모델 설정
    model_name = st.sidebar.selectbox(
        "요약 모델 선택:",
        ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-base"],
        # TODO: 요약모델 최종 확인 필요, bart-large-cnn 빼고 정한게 있나..?
        index=0  # facebook/bart-large-cnn을 기본값으로 설정
    )
    
    # 메인 영역 설정
    col1, col2 = st.columns(2)
    
    # 요약 버튼
    if st.sidebar.button("요약하기", type="primary"):
        if text_input:
            with st.spinner("요약 중..."):
                # 실제 요약 및 어텐션 스코어 계산
                summary, attention_scores = get_summary_and_attention(text_input, model_name)
                
                # session_state에 결과 저장
                st.session_state.summary = summary
                st.session_state.attention_scores = attention_scores
                st.session_state.text_input = text_input
                
                # ROUGE와 BERTScore 계산 및 표시
                # ROUGE 점수 계산
                rouge_scores = calculate_rouge(summary, text_input)
                
                # BERTScore 계산
                bert_score_value = calculate_bert_score(summary, text_input)
                
                # 사이드바에 평가 점수 표시
                st.sidebar.divider()
                st.sidebar.header("평가 결과")
                
                # ROUGE 점수
                st.sidebar.write("#### ROUGE 점수")
                col1_rouge, col2_rouge = st.sidebar.columns(2)
                with col1_rouge:
                    st.metric("ROUGE-1", f"{rouge_scores['rouge-1']['f']:.3f}")
                    st.metric("ROUGE-2", f"{rouge_scores['rouge-2']['f']:.3f}")
                with col2_rouge:
                    st.metric("ROUGE-L", f"{rouge_scores['rouge-l']['f']:.3f}")
                
                # BERT 점수
                st.sidebar.write("#### BERT 점수")
                st.sidebar.metric("BERTScore", f"{bert_score_value:.3f}")

    # session_state에 저장된 결과가 있을 때만 표시
    if 'summary' in st.session_state:
        with col1:
            st.header("요약 결과")
            view_mode = st.segmented_control(
                label="보기 모드를 선택하세요",
                options=["전체 문장", "특정 주제"],
                default="전체 문장"
            )
            
            if view_mode == "특정 주제":
                st.info("""
                💡 **특정 주제 모드 사용 방법**
                - 여기에 scatter plot 들어갈 예정.
                1. 아래의 요약 문장들을 클릭하세요.
                2. 클릭한 요약 문장과 관련된 원본 문장들이 오른쪽에 하이라이트되어 표시됩니다.
                3. 클릭한 문장에 대한 상세 정보가 아래에 표시됩니다.
                4. 같은 문장을 다시 클릭하면 상세 정보가 닫힙니다.
                """)
                
                # 요약 문장을 버튼으로 표시
                summary_sentences = st.session_state.summary.split('. ')
                
                for i, sent in enumerate(summary_sentences):
                    if sent:  # 빈 문장 제외
                        if st.button(f"{sent}.", key=f"topic_summary_sent_{i}"):
                            # 현재 버튼이 이미 활성화되어 있다면 닫기
                            if st.session_state.popup_states.get(i, False):
                                st.session_state.popup_states[i] = False
                            else:
                                # 다른 모든 팝업은 닫고 현재 선택한 것만 열기
                                st.session_state.popup_states = {k: False for k in st.session_state.popup_states.keys()}
                                st.session_state.popup_states[i] = True
                            st.session_state.selected_sentence = sent
                        
                        # 팝업 표시
                        # TODO: Brushing 재요약 부분 여기 markdown으로 해결하기
                        if st.session_state.popup_states.get(i, False):
                            st.markdown(
                                f"""
                                <div class="summary-box">
                                    <h4>상세 정보</h4>
                                    <p>• 핵심 키워드: [관련 키워드들]</p>
                                    <p>• 관련 문맥: {sent}와 관련된 추가적인 설명</p>
                                    <p>• 연관 개념: [관련된 주요 개념들]</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

            if view_mode == "전체 문장":
                st.info("""
                💡 **전체 문장 모드 설명서**
                - 옆에 나온 원문 텍스트의 색깔은 요약 모델이 어디를 집중했는지 시각화한 모습입니다.
                """)
                # 일반 텍스트를 네모 박스 안에 표시
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px;">
                        {st.session_state.summary}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                pass

        with col2:
            st.header("원본 텍스트")
            if view_mode == "전체 문장":
                # 기존의 attention score 시각화
                html_content = create_attention_html(st.session_state.text_input, st.session_state.attention_scores)
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px;">
                        {html_content}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:  # "특정 주제" 모드
                if st.session_state.selected_sentence:
                    # 원본 텍스트를 문장 단위로 분리
                    original_sentences = st.session_state.text_input.split('. ')
                    
                    # 선택된 요약 문장과 관련된 원본 문장 찾기
                    html_content = ""
                    for orig_sent in original_sentences:
                        if orig_sent:  # 빈 문장 제외
                            # 여기서는 임시로 랜덤하게 관련성 표시
                            is_related = np.random.random() > 0.7
                            if is_related:
                                html_content += f'<span style="background-color: rgba(255, 255, 0, 0.5)">{orig_sent}.</span> '
                            else:
                                html_content += f'{orig_sent}. '
                    
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px;">
                            {html_content}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.info("요약 문장을 클릭하면 관련된 원본 문장이 하이라이트됩니다.")
            
    else:
        st.sidebar.error("텍스트를 입력하거나 파일을 업로드해주세요.")

    # 스타일 설정
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
            margin-top: 20px;
        }
        .uploadedFile {
            border: 1px solid #ccc;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()