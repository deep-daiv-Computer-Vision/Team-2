from flask import Flask, request, jsonify
from flask_cors import CORS  # pip install flask-cors
from mainrun import summarize_and_visualization, brushing_and_resummarize

app = Flask(__name__)
CORS(app)  # CORS 활성화

@app.route('/summarize', methods=['POST'])
def summarize():
    # 텍스트 직접 입력과 파일 업로드 두 가지 경우 처리
    text_to_summarize = None
    selected_model = request.form.get('model')

    # 파일 업로드 케이스 체크
    if 'file' in request.files:
        file = request.files['file']
        if file and file.filename.endswith('.txt'):
            text_to_summarize = file.read().decode('utf-8')
    
    # 직접 텍스트 입력 케이스 체크
    elif 'text' in request.form:
        text_to_summarize = request.form.get('text')

    # 텍스트가 없는 경우 에러 반환
    if not text_to_summarize:
        return jsonify({'error': '텍스트를 입력하거나 파일을 업로드해주세요.'}), 400

    # 실험 실행
    try:
        experiment_results = summarize_and_visualization(text_to_summarize, selected_model)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # 실험 결과를 반환합니다.
    return jsonify({'experiments': experiment_results})

@app.route('/resummarize', methods=['POST'])
def resummarize():
    data = request.json
    modified_text = data.get('text')
    selected_model = data.get('model')

    if not modified_text:
        return jsonify({'error': '텍스트를 입력해주세요.'}), 400

    try:
        experiment_results = brushing_and_resummarize(modified_text, selected_model)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # 실험 결과를 반환합니다.
    return jsonify({'experiments': experiment_results})

if __name__ == '__main__':
    app.run(debug=True)