import datetime
import os.path

import numpy
from flask import Flask, request, jsonify
from recognition.middleware import ModelManager

app = Flask(__name__)
model_path = "storage/model.h5"
images_folder = "storage/client_images/"


def format_datetime(dt):
    return str(dt).replace(" ", "").replace(".", "").replace(":", "")


@app.route('/camera/process/', methods=["POST"])
def hello_world():  # put application's code here
    data = request.files["imageFile"]
    img_path = os.path.join(images_folder, f"{format_datetime(datetime.datetime.now())}_img.jpg")
    data.save(img_path)
    manager = ModelManager(model_path, img_path)
    return manager.process()


if __name__ == '__main__':
    app.run(host="192.168.88.33")
