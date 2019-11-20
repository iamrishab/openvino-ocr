import config
if config.INFERENCE_ENGINE_TYPE == 'opencv':
    import text_detection_cv as text_detection
    import text_recognition_cv as text_recognition
else:
    import text_detection_ie as text_detection
    import text_recognition_ie as text_recognition
import cv2
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, type=str, help="path to input image")
args = vars(ap.parse_args())

def  main():
    try:
        image = cv2.imread(args["image"])
        td = text_detection.PixelLinkDecoder()
        tr = text_recognition.TextRecognizer()

        image_show, bounding_rects = td.inference(image)
        texts = tr.inference(image, bounding_rects)    

        print('Result:', texts)

        cv2.imshow('Detected text', image_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()

