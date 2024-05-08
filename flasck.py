from flask import Flask, request, jsonify
from fastai.vision.all import *
import pathlib
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
app = Flask(__name__)
learn = load_learner('modeldatesF.pkl')
learn2 = load_learner('modeldatesFquality.pkl')
labels = learn.dls.vocab
labels2 = learn2.dls.vocab


@app.route('/predict', methods=['POST'])
def predict():
    # Check if the POST request has a file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Load the image and make prediction
    img = PILImage.create(file)
    dates_type, idx, probs = learn.predict(img)

    if dates_type == "skary web":
        dates_type2, idx2, probs2 = learn2.predict(img)
      
            # Save the image to the upload folder
        return jsonify({'save': 'Image saved successfully.','result1': str(dates_type), 'accruracy': float(probs[idx]), 'result2': str(dates_type2),
                        'accruracy2': float(probs[idx])})

    else:
        return jsonify({'save': 'Image saved successfully.','result': str(dates_type), 'accuracy': float(probs[idx])})


if __name__ == '__main__':
    app.run(debug=True)
