import streamlit as st
import pandas as pd
import numpy as np

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

def main():
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
            "요약할 파일을 업로드하세요",
            type=['txt', 'pdf', 'docx'],
            help="최대 200MB까지 업로드 가능합니다."
        )
        
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
            height=300,
            placeholder="여기에 요약할 텍스트를 입력하세요..."
        )

    # 구분선 추가
    st.sidebar.divider()
    
    # 모델 평가 섹션
    st.sidebar.header("Model Evaluation")
    
    # 모델 설정
    model_options = st.sidebar.multiselect(
        "평가 모델 선택:",
        ["ROUGE", "BERT", "기타"],
        default=["ROUGE", "BERT"]
    )
    
    # 메인 영역 설정
    col1, col2 = st.columns(2)
    
    # 요약 버튼
    if st.sidebar.button("요약하기", type="primary"):
        if text_input:
            with st.spinner("요약 중..."):
                # 예시 요약 및 어텐션 점수 (실제 구현 시에는 모델에서 계산된 값을 사용)
                summary = "이것은 요약된 텍스트입니다. 주요 내용이 포함됩니다."
                # 랜덤한 어텐션 점수 생성 (예시용)
                attention_scores = np.random.random(len(text_input.split()))
                
                with col1:
                    st.header("요약 결과")
                    st.write(summary)
                    
                    # ROUGE와 BERT 점수를 위한 컬럼 생성
                    score_col1, score_col2 = st.columns(2)
                    
                    with score_col1:
                        if "ROUGE" in model_options:
                            st.write("#### ROUGE 점수")
                            st.metric("ROUGE-1", "0.45")
                            st.metric("ROUGE-2", "0.32")
                            st.metric("ROUGE-L", "0.40")
                    
                    with score_col2:
                        if "BERT" in model_options:
                            st.write("#### BERT 점수")
                            st.metric("BERT Score", "0.78")

                with col2:
                    st.header("원본 텍스트 (Attention Score 시각화)")
                    # Attention score를 적용한 HTML 생성
                    html_content = create_attention_html(text_input, attention_scores)
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px;">
                            {html_content}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
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