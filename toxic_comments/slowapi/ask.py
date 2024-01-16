from flask import Flask, render_template, request
#from toxic_comments.predict_model import predict_user_input_hosting

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('1.html')

@app.route('/upload', methods=['POST'])
def upload():
    text_input = request.form.get('text_input')
    
    file = request.files['file_input']
    file.save(f"uploads/{file.filename}")  # Save the uploaded file to a folder named 'uploads'

    # Do something with the text and file, for example, print them
    print(f"Text Input: {text_input}")
    print(f"File Uploaded: {file.filename}")
    predict_user_input_hosting(text_input)

    return "Form submitted successfully!"

if __name__ == '__main__':
    app.run(debug=True)
