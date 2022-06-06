from socket import SO_LINGER
from flask import Flask, flash, request, jsonify, redirect, url_for
from detection import nearestClass, getNeighbors, dataset, results
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
from tempfile import TemporaryFile
import os
from collections import defaultdict
from werkzeug.utils import secure_filename

path = os.path.abspath('.')
UPLOAD_FOLDER = path + './upload'
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return "bismillah protel"


@app.route("/deteksi", methods=['GET', 'POST'])
def deteksi():
    #mendapatkan audio dan menyimpannya
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename("sample.wav")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    #memanggil model deteksi genre musik
    directory = os.path.abspath('.') 
    (rate,sig) = wav.read(directory + './upload/sample.wav')
    mfcc_feat = mfcc(sig, rate, winlen = 0.010, appendEnergy = False)
    covariance = np.cov(np.matrix.transpose(mfcc_feat))
    mean_matrix = mfcc_feat.mean(0)
    feature = (mean_matrix, covariance, 0)

    #membuat prediksi genre
    pred = nearestClass(getNeighbors(dataset, feature, 5))

    #extractfeature(directory + '/upload/sample.wav')

    #mengirimkan prediksi dalam json
    #print('genre:', results[pred], '\n', 'music recommendation:', '\n', Similiarity('sample.wav'))
    return jsonify({'genre': results[pred]})
    #'music recommendation': Similiarity

if __name__ == '__main__':
    app.run(debug=True)