# 'opencv' or 'openvino'
INFERENCE_ENGINE_TYPE = 'opencv'

PATH_TEXT_RECOGNITION_MODEL_XML = 'text_recognition/text-recognition-0012.xml'
PATH_TEXT_RECOGNITION_MODEL_BIN = 'text_recognition/text-recognition-0012.bin'

PATH_TEXT_DETECTION_MODEL_XML = 'text_detection/text-detection-0003.xml'
PATH_TEXT_DETECTION_MODEL_BIN = 'text_detection/text-detection-0003.bin'

PATH_TO_INFERENCE_ENGINE = '/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'
# 'flask' or 'wsgi'
SERVER_TYPE = 'flask'

IP = '0.0.0.0'
PORT = 9005

HITURL = 'http://127.0.0.1:9001/ocr'
URL = 'http://'+IP+':'+str(PORT)+'/ocr'
