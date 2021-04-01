import os
import cv2
from src.dist_class import Distortion_Classification
from flask import Flask, request, make_response, render_template

app = Flask(__name__)

# set paths to upload folder
APP_ROOT = os.path.dirname(os.path.abspath('POC_Flask/models'))
app.config['IMAGE_UPLOADS'] = os.path.join(APP_ROOT, 'static')


@app.route("/face-distortion-classifier", methods=["GET", "POST"])
def classify_image():
    if request.method == "POST":
        # read and upload resized files to folder
        image = request.files['input_file']
        filename = image.filename
        file_path = os.path.join(app.config["IMAGE_UPLOADS"], filename)
        img = cv2.imread(file_path)

        # classify image
        dist_class = Distortion_Classification(img)
        results = dist_class.results
        main_label = results[0]['Main Label']



        # display prediction and image
        return render_template("upload.html", image_path=filename, prediction='Prediction: ' + main_label)
    return render_template("upload.html", image_path='landing_page_pic.jpg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, threaded=False, port=8000)