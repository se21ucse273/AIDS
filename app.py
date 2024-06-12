# app.py
from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from Test import convert_pdf_to_txt

app = Flask(__name__, static_folder='static')

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/process', methods=['POST'])
def process_pdf():
    pdf_file = request.files['pdf_file']
    txt_filename = request.form['txt_filename']

    if pdf_file.filename == '':
        return redirect(request.url)

    if pdf_file and allowed_file(pdf_file.filename):
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
        pdf_file.save(pdf_path)
        dir_file = os.path.join(app.config["OUTPUT_FOLDER"], txt_filename)
        file_path = dir_file+'.txt'
        summarized_text = convert_pdf_to_txt(pdf_path, file_path)
        with open(file_path, 'w') as f:
            f.write(summarized_text)  # You can also write something here if you want to add initial content

        #os.system(f'python Test.py {pdf_path} {file_path}')
        return redirect(url_for('result', txt_filename=f"{txt_filename}.txt"))

@app.route('/result')
def result():
    txt_filename = request.args.get('txt_filename')
    return render_template('result.html', txt_filename=txt_filename)

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
