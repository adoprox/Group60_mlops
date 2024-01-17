from flask import Flask, render_template, request
from toxic_comments.predict_model import predict_user_input_hosting
from omegaconf import OmegaConf
import pandas as pd


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('1.html')

@app.route('/upload', methods=['POST'])
def upload():

    config = OmegaConf.load("toxic_comments/models/config/default.yaml")  # Update with the actual path
    text_input = request.form.get('text_input')
    config.text = text_input

    #file = request.files['file_input']
    #file.save(f"uploads/{file.filename}")  # Save the uploaded file to a folder named 'uploads'

    print(f"Text Input: {text_input}")
    #print(f"File Uploaded: {file.filename}")
    df = predict_user_input_hosting(config)
    return render_template('index.html', table=df.to_html(classes='table table-striped'))

if __name__ == '__main__':
    app.run(debug=True)
