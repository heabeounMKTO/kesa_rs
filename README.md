# kesa (កេសា)
tool for auto labeling, conversion and augmentation for darknet YOLO (and other annotation formats?).<br>
kesa comes with a few binaries.
|name|explanation|
|---|---|
|kesa_al| for auto labeling, comes with onnx (ort) and torch (tch-rs) backends|
|kesa_l2y| for converting annotations to yolo txt format|
|kesa_split| for separating images/annotations to train, val, test batches.|
|kesa_aug| creates image augmentations from given labels and images|


# external dependencies
## onnxruntime 
- build/download library [onnxruntime](https://github.com/microsoft/onnxruntime)
- add to path (in this example this is the cuda version):
  ```bash
  export LD_LIBRARY_PATH=/path/to/onnxruntime-linux-x64-cuda-1.17.1/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
  ```
## tch-rs 
*wip*

# building 
make sure you have the dependencies listed above <br>
to build , run:
```bash
cargo build --release
```
