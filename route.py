from flask import Flask, request, jsonify, send_file, send_from_directory
import os
from werkzeug.utils import secure_filename
import traceback
from mainrun import exe_by_sentences, resummarize_with_sentence
from flask_cors import CORS
import torch

app = Flask(__name__)
CORS(app)

# GPU 사용 가능 여부 확인 및 device 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ALLOWED_EXTENSIONS = {'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        # Parse JSON input
        if not request.is_json:
            return jsonify({"error": "Request must be in JSON format"}), 400

        data = request.get_json()
        select_model = data.get("select_model")
        text = data.get("text")
        file = request.files.get("file")

        # Validate inputs
        if not select_model:
            return jsonify({"error": "select_model is required"}), 401

        if (text and file) or (not text and not file):
            return jsonify({"error": "Either text or file must be provided, but not both"}), 402

        # Process text or file
        if file:
            if not allowed_file(file.filename):
                return jsonify({"error": "Only .txt files are allowed"}), 403

            filename = secure_filename(file.filename)
            filepath = os.path.join("/tmp", filename)
            file.save(filepath)

            with open(filepath, "r") as f:
                text = f.read()

            os.remove(filepath)

        # Ensure text is not empty
        if not text.strip():
            return jsonify({"error": "Text content is empty"}), 404

        # Call exe_by_sentences function
        segments, concat_indices, batch_summaries, batch_importances, evaluation_results, visualize_pth = exe_by_sentences(text)

        print("batch_importances structure:", type(batch_importances))
        print("first element structure:", type(batch_importances[0]))
        if len(batch_importances) > 0 and len(batch_importances[0]) > 0:
            print("innermost element structure:", type(batch_importances[0][0]))

        # Prepare response
        response = {
            "batch_summaries": batch_summaries,
            "batch_importances": [[[float(z) for z in y] for y in x] for x in batch_importances],
            "evaluation_results": {
                k: float(v) for k, v in evaluation_results.items()
            },
            "segments": segments,
            "concat_indices": concat_indices
        }

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/resummarize', methods=['POST'])
def resummarize():
    try:
        # Parse JSON input
        if not request.is_json:
            return jsonify({"error": "Request must be in JSON format"}), 400

        data = request.get_json()
        full_text = data.get("full_text")
        target_text = data.get("target_text")

        # Validate inputs
        if not full_text or not target_text:
            return jsonify({"error": "Both full_text and target_text are required"}), 400

        # Call resummarize_with_sentece function
        summary = resummarize_with_sentence(full_text, target_text)

        return jsonify({"summary": summary})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
