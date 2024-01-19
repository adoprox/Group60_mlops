from flask import Flask, render_template, request
from toxic_comments.predict_model import load_model, predict
from omegaconf import OmegaConf
from hydra import initialize, compose

pred_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

app =  Flask(__name__)


def read_text_file(file_stream):
    try:
        # Use TextIOWrapper to handle the file stream
        file_wrapper = TextIOWrapper(file_stream, encoding='utf-8')
        lines = file_wrapper.readlines()
        lines = [line.strip() for line in lines]
        return lines
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def load_config():  
    """Decorator to cache loaded config. Avoids errors of hydra complaining about reinitialization."""
    with initialize("../models/config"):
        a = compose( config_name="default.yaml",
            overrides=["predict.checkpoint_path=models/production/model.ckpt"],)
        return a

def load_model_decorator(config):
    return load_model(config)

def run_model(text_input):
    packed_model = load_config()
    token, model, dev = load_model_decorator(packed_model)
    pred = predict(text_input, token, model, dev)
    return pred


@app.route("/")
def index():
    
    return render_template("home.html")

@app.route("/upload", methods=["POST"])
def upload():
    text_input = request.form.get("text_input")
    user_text = text_input
    final_op=run_model(user_text)
    result = zip(pred_labels, final_op[0][0])
    results_web = [f"* {label}: {p*100:.1f}%" for label, p in result]
    print(type(results_web))
    return render_template("submit.html", results_web=results_web)

@app.route("/file_upload", methods=["POST"])
def file_upload():
    if 'file_input' not in request.files:
        return render_template("form.html", error="No file part")

    file = request.files['file_input']

    if file.filename == '':
        return render_template("form.html", error="No selected file")

    comments_list = read_text_file(file.stream)
    file_op = run_model(comments_list)

    results_formatted = []
    for line, predicted in zip(comments_list, file_op):
        labels = pred_labels
        probabilities = predicted[0]
        results_formatted.append({"input_line": line, "labels": labels, "probabilities": probabilities})

    return render_template("form.html", results_formatted=results_formatted)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)