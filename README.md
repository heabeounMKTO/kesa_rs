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
todo

# building 
make sure you have the dependencies listed above <br>
to build , run:
```bash
cargo build --release
```
