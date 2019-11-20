import config
if config.INFERENCE_ENGINE_TYPE == 'opencv':
    import text_detection_cv as text_detection
    import text_recognition_cv as text_recognition
else:
    import text_detection_ie as text_detection
    import text_recognition_ie as text_recognition
import cv2
from flask import Flask, Response, request
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
import numpy as np
import json

app = Flask(__name__)

td = text_detection.PixelLinkDecoder()
tr = text_recognition.TextRecognizer()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)  

# route http posts to this method
@app.route('/ocr', methods=['POST', 'GET'])
def main():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)

    # decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    image_show, bounding_rects = td.inference(image)
    texts = tr.inference(image, bounding_rects)    

    result = dict(zip(texts, bounding_rects))

    response = json.dumps(result, cls=NumpyEncoder)

    return Response(response=response, status=200, mimetype="application/json")


if __name__ == '__main__':
    if config.SERVER_TYPE == 'wsgi':
        from gevent.pywsgi import WSGIServer
        WSGIServer((config.IP, config.PORT), app).serve_forever()
    elif config.SERVER_TYPE == 'flask':
        app.run(config.IP, config.PORT, debug=True, threaded=True)
    else:
        raise Exception('server type not found')
