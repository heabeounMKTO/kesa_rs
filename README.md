# kesa (កេសា)
tool for auto labeling, conversion and augmentation for darknet YOLO (and other annotation formats)

# dependencies
because kesa depends on onnxruntime to run the auto labeling function, you need to include libonnxruntime on your path
<br>
1. download the onnxruntime libraries [here](https://github.com/microsoft/onnxruntime/releases)

2. add the downloaded libs to path

```bash
vim ~/.bashrc
# Add the path of ONNXRUntime lib
export LD_LIBRARY_PATH=/path/to/onnxruntime-linux-x64-gpu-1.17.1/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
source ~/.bashrc
```

# building kesa binaries
following the unix principle , each of the kesa binaries does one thing only and it does it very well.
to build them all run:
```bash
cargo build --release
```
# what does kesa binaries do
## kesa_al
labels images in a folder using yolo models exported in onnx format
(currently only works with yolov7 models , but patch for v9 is coming s00n)

## kesa_aug
creates variations from labeled images

## kesa_l2y

converts labelme annotations to yolo annotations
