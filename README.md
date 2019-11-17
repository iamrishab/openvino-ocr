# openvino-ocr

Openvino toolkit allows developers to deploy pre-trained deep learning models through a high-level C++ Inference Engine API integrated with application logic. 

This open source version includes two components, namely Model Optimizer and Inference Engine, as well as CPU, GPU and heterogeneous plugins to accelerate deep learning inferencing on Intel(R) CPUs and Intel(R) Processor Graphics.

## Required Changes
After installing OpenVino, change the below config based on your installation folder.
PATH_TO_INFERENCE_ENGINE = '/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'

## To run the code (Python 3):
$ python3 main.py --image path/to/image.jpg

## Repository components:
* [Inference Engine](https://software.intel.com/en-us/articles/OpenVINO-InferEngine)
* [Model Optimizer](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer)

## Documentation
* [OpenVINO™ Release Notes](https://software.intel.com/en-us/articles/OpenVINO-RelNotes)
* [Inference Engine build instructions](inference-engine/README.md)
* [Get Started with Deep Learning Deployment Toolkit on Linux*](get-started-linux.md)
* [Introduction to Deep Learning Deployment Toolkit](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Introduction.html)
* [Inference Engine Developer Guide](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html)
* [Model Optimizer Developer Guide](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
