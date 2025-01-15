from flask import Flask, request, jsonify
from transformers import pipeline
from experiment import summarize_and_visualization
from main import summarize_and_visualization

app = Flask(__name__)

@app.route('/summarize', methods=['POST'])
def summarize():
    # 프론트엔드로부터 '요약할 텍스트'와 '선택할 모델'을 받습니다.
    data = request.json
    text_to_summarize = data.get('text')
    selected_model = data.get('model')

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
        experiment_results = summarize_and_visualization(modified_text, selected_model)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # 실험 결과를 반환합니다.
    return jsonify({'experiments': experiment_results})

if __name__ == '__main__':
    app.run(debug=True)