import cv2
from openvino.inference_engine import IENetwork, IEPlugin
import numpy as np
import config

class TextRecognizer:
    def __init__(self):
        self.symbols = "0123456789abcdefghijklmnopqrstuvwxyz#"
        plugin = IEPlugin(device='CPU')
        net = IENetwork(model=config.PATH_TEXT_RECOGNITION_MODEL_XML, weights=config.PATH_TEXT_RECOGNITION_MODEL_BIN)
        plugin.add_cpu_extension(config.PATH_TO_INFERENCE_ENGINE)
        
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            print('thing')
        
        print("Preparing input blobs")
        self.input_blob = next(iter(net.inputs))
        self.out_blob = next(iter(net.outputs))
        
        print("Loading model to the plugin")
        self.exec_net = plugin.load(network=net)
        del net
        
    def process(self, img_source):
        img = img_source.copy()
        input_width = 120
        input_height = 32
        img_height = img.shape[0]
        img_width = img.shape[1]
        blob = cv2.dnn.blobFromImage(img, 1.0, (input_width, input_height))
        res = self.exec_net.infer({self.input_blob: blob})
        res = res[self.out_blob]
        return res

    def ctc_decoder(self, data):
        result = ""
        prev_pad = False
        num_classes = len(self.symbols)
        for i in range(data.shape[0]):
            symbol = self.symbols[np.argmax(data[i])]
            if symbol != self.symbols[-1]:
                if len(result) == 0 or prev_pad or (len(result) > 0 and symbol != result[-1]):
                    prev_pad = False
                    result = result + symbol
            else:
                prev_pad = True
        return result

    def inference(self, img, bbrect):
        texts = []
        for x, y, w, h in bbrect:
            img_gray = cv2.cvtColor(img[y:y+h, x:x+w, :], cv2.COLOR_BGR2GRAY)
            result = self.process(img_gray)
            text = self.ctc_decoder(result).strip()
            if text:
                cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1)
                texts.append(text)
        return texts


