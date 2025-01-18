from flask import Flask, request, jsonify
from flask_cors import CORS
from mainrun import summarize_and_visualization, brushing_and_resummarize

app = Flask(__name__)
CORS(app)

@app.route('/summarize', methods=['POST'])
def summarize():
    # 프론트엔드로부터 '.txt' 파일과 '선택할 모델'을 받습니다.
    file = request.files.get('file')
    selected_model = request.form.get('model')

    if not file or not file.filename.endswith('.txt'):
        return jsonify({'error': '유효한 .txt 파일을 업로드하세요.'}), 400

    # 파일 내용을 읽습니다.
    text_to_summarize = file.read().decode('utf-8')

    # 실험 실행
    try:
        experiment_results = summarize_and_visualization(text_to_summarize, selected_model)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # 실험 결과를 반환합니다.
    return jsonify({'experiments': experiment_results})

@app.route('/resummarize', methods=['POST'])
def resummarize():
    # 프론트엔드로부터 '수정된 텍스트'와 '선택할 모델'을 받습니다.
    data = request.json
    modified_text = data.get('text')
    selected_model = data.get('model')

    # 실험 실행
    try:
        experiment_results = brushing_and_resummarize(modified_text, selected_model)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # 실험 결과를 반환합니다.
    return jsonify({'experiments': experiment_results})

if __name__ == '__main__':
    app.run(debug=True)