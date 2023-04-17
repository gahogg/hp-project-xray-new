from flask import Flask, render_template, request, redirect, url_for
import os
from model_stuff import get_img_with_heatmap
from tensorflow.keras.models import load_model
from PIL import Image


app = Flask(__name__)

HEATMAP_FOLDER = 'static/heatmaps/'
EXAMPLES_FOLDER = 'static/examples/'

loaded_model = load_model('model/')

app.config['HEATMAP_FILEPATH'] = 'static/images/zz.jpg'
app.config['DIAGNOSIS'] = 'N/A'
app.config['CONFIDENCE'] = 'N/A'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    filename = request.form.get('file')
    example_filepath = os.path.join(EXAMPLES_FOLDER, filename)

    image_with_heatmap, (class_name, confidence) = get_img_with_heatmap(example_filepath, 
                                                                        0.3, loaded_model)
    im = Image.fromarray(image_with_heatmap)
    heatmap_path = os.path.join(HEATMAP_FOLDER, filename)
    im.save(heatmap_path)

    app.config['HEATMAP_FILEPATH'] = heatmap_path
    app.config['DIAGNOSIS'] = class_name
    app.config['CONFIDENCE'] = confidence

    return redirect(url_for('play'))


@app.route('/play')
def play():
    files = os.listdir(EXAMPLES_FOLDER)
    return render_template('play.html', filepath=app.config['HEATMAP_FILEPATH'],
                                        diagnosis=app.config['DIAGNOSIS'],
                                        confidence=app.config['CONFIDENCE'],
                                        files=files)


if __name__ == '__main__':
    app.run(debug=False)
