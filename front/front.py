import streamlit as st
import numpy as np
import requests

# TODO: 백엔드와 연결하는 작업 필요

# 페이지 설정을 가장 먼저 호출
st.set_page_config(
    page_title="XAI Summarization",
    page_icon="🤖", layout="wide"
    )

def create_attention_html(text, attention_scores):
    """텍스트에 attention score를 적용하여 HTML로 변환"""
    words = text.split()
    # attention_scores의 길이가 words의 길이와 다른 경우 처리
    if len(attention_scores) != len(words):
        # 길이를 맞추기 위해 attention_scores를 리샘플링
        attention_scores = np.interp(
            np.linspace(0, 1, len(words)),
            np.linspace(0, 1, len(attention_scores)),
            attention_scores
        )
    
    html = ""
    for word, score in zip(words, attention_scores):
        # score는 이미 0-1 사이의 값으로 정규화되어 있음
        html += f'<span style="background-color: rgba(255, 0, 0, {score:.2f})">{word}</span> '
    return html

def get_summary_and_attention(text, model_name):
    """텍스트 요약 및 어텐션 스코어 계산"""
    try:
        data = {'select_model': model_name}
        
        # 파일 업로드인 경우
        if isinstance(text, bytes) or hasattr(text, 'read'):
            files = {
                'file': ('input.txt', text if isinstance(text, bytes) else text.read(), 'text/plain')
            }
            response = requests.post(
                "http://localhost:5000/summarize",
                files=files,
                data=data
            )
        # 직접 텍스트 입력인 경우
        else:
            data['text'] = text
            response = requests.post(
                "http://localhost:5000/summarize",
                json=data
            )

        if response.status_code == 200:
            result = response.json()
            experiment = result['experiments'][0]
            token_importance = np.array(experiment.get('token_importance', []))
            
            if len(token_importance) == 0:
                return experiment.get('summary'), np.zeros(len(text.split()))
                
            # 정규화 시 예외 처리
            token_max = token_importance.max()
            token_min = token_importance.min()
            if token_max == token_min:
                normalized_importance = np.full_like(token_importance, 0.5)
            else:
                normalized_importance = (token_importance - token_min) / (token_max - token_min)
                
            return experiment.get('summary'), normalized_importance
        else:
            st.error(f"API 오류: {response.status_code}")
            return None, None
    except Exception as e:
        st.error(f"연결 오류: {str(e)}")
        return None, None

def calculate_rouge(summary, reference):
    """ROUGE 점수 계산"""
    try:
        response = requests.post(
            "http://localhost:5000/calculate-rouge",
            json={
                "summary": summary,
                "reference": reference
            }
        )
        if response.status_code == 200:
            result = response.json()
            # route.py에서 반환하는 형식에 맞춰 값을 추출
            return {
                'rouge1': result['rouge-1']['f'],
                'rouge2': result['rouge-2']['f'],
                'rougeL': result['rouge-l']['f']
            }
        else:
            st.error(f"ROUGE 계산 오류: {response.status_code}")
            return {
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0
            }
    except Exception as e:
        st.error(f"연결 오류: {str(e)}")
        return {
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0
        }

def calculate_bert_score(summary, reference):
    """BERTScore 계산"""
    try:
        response = requests.post(
            "http://localhost:5000/calculate-bertscore",
            json={
                "summary": summary,
                "reference": reference
            }
        )
        if response.status_code == 200:
            return response.json()["score"]  # float 값으로 반환됨
        else:
            st.error(f"BERTScore 계산 오류: {response.status_code}")
            return 0.0
    except Exception as e:
        st.error(f"연결 오류: {str(e)}")
        return 0.0

def get_resummarize(text, model_name):
    """선택된 문장에 대한 재요약 결과 가져오기"""
    try:
        response = requests.post(
            "http://localhost:5000/resummarize",
            json={
                "text": text,
                "model": model_name
            }
        )
        if response.status_code == 200:
            result = response.json()
            # 첫 번째 실험 결과만 반환
            return result['experiments'][0] if result['experiments'] else None
        else:
            st.error(f"재요약 API 오류: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"연결 오류: {str(e)}")
        return None

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

    st.title("🔮💯시험 공부 벼락치기 시트 만들기")
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
                    st.metric("ROUGE-1", f"{rouge_scores['rouge1']:.3f}")
                    st.metric("ROUGE-2", f"{rouge_scores['rouge2']:.3f}")
                with col2_rouge:
                    st.metric("ROUGE-L", f"{rouge_scores['rougeL']:.3f}")
                
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
                # TODO: 이곳에 scatter plot 들어갈 예정, 연결바람
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
                        if st.session_state.popup_states.get(i, False):
                            # 재요약 결과 가져오기
                            resummarize_result = get_resummarize(sent, model_name)
                            
                            if resummarize_result:
                                st.markdown(
                                    f"""
                                    <div class="summary-box">
                                        <h4>Brushing Resummarize ✨</h4>
                                        <p>• 재요약 결과: {resummarize_result['summary']}</p>
                                        <p>• 관련 문맥: {sent}</p>
                                    """,
                                    unsafe_allow_html=True
                                )
                            else:
                                st.error("재요약 결과를 가져오는데 실패했습니다.")

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
                    
                    # TODO: 선택된 요약 문장과 관련된 원본 문장 (clustering에서 활용) 연결 (현재는 random하게 표시 중)
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