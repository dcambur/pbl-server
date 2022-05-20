import tensorflow
from flask import jsonify
from tensorflow import keras


class ModelManager:
    __TREAT = "TREAT"
    __WAIT = "WAIT"
    IMG_SIZE = (180, 180)

    def __init__(self, path_to, img_path):
        self.path_to = path_to
        self.img_path = img_path
        self.__model = keras.models.load_model(path_to)

    def to_json(self, pred_scores):
        print(pred_scores)
        return jsonify(
            {
                "health_predict": str(pred_scores[0]),
                "sick_predict": str(pred_scores[1]),
                "action": self.__WAIT if pred_scores[0] > pred_scores[1] else self.__TREAT
            }
        )

    def process(self):
        img = keras.preprocessing.image.load_img(self.img_path, target_size=self.IMG_SIZE)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tensorflow.expand_dims(img_array, 0)  # Create batch axis

        return self.to_json(self.__model.predict(img_array)[0])
