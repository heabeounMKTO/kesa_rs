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
currently `kesa_al` uses either torch(tch-rs) or onnxruntime(ort) to label images,
you can compile with either or both. you just need to download and link the libraries.
## onnxruntime 
- build/download library [onnxruntime](https://github.com/microsoft/onnxruntime)
- add to your ~/.zshrc or ~/.bashrc:
  ```bash
  export LD_LIBRARY_PATH=/path/to/onnxruntime-linux-x64-cuda-1.17.1/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
  ```
## tch-rs 
- download libtorch from [pytorch](https://pytorch.org/) site
- add to your ~/.zshrc or ~/.bashrc:
  ```bash
  export LIBTORCH=/media/hbpopos/penisf/libtorch-cxx11-abi-shared-with-deps-2.2.0+cu121/libtorch
  export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
  ```
- alternative options for installing `tch-rs`  can be found [here](https://github.com/LaurentMazare/tch-rs?tab=readme-ov-file#libtorch-manual-install)

# building 
make sure you have the dependencies listed above <br>
to build , run:
```bash
cargo build --release
```
