# DLA Compiler

### Layers and features support

|Layer &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;|Feature &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;|FP16 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;|INT8 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;|
|-----------|---------------|-------|-------|
|**Convolution**||&#10004;|&#10004;|
||Dilation|&#10004;|&#10004;|
||Winograd|&#10004;|Not implemented in SW|
|**Deconvolution**||&#10004;|&#10004;|
||With padding|Not implemented in SW|Not implemented in SW|
||Winograd|Not implemented in SW|Not implemented in SW|
|**Fully Connected**||&#10004;|&#10004;|
||Winograd|Not implemented in SW|Not implemented in SW|
|**Group Convolution**||&#10004;|Not implemented in SW|
||Winograd|&#10004;|Not implemented in SW|
|**Pooling**||&#10004;|&#10004;|
||Max|&#10004;|&#10004;|
||Min|&#10004;|&#10004;|
||Avg|&#10004;|&#10004;|
||Inclusive padding|&#10004;|&#10004;|
||Exclusive padding|Not supported in HW| Not supported in HW|
|**Activation**||||
||Bias|&#10004;|&#10004;|
||BatchNorm|&#10004;|&#10004;|
||Scale|&#10004;|&#10004;|
||Sigmoid|&#10004;|Not implemented in SW|
||Tanh|&#10004;|Not implemented in SW|
||EltWise SUM|&#10004;|&#10004;|
||EltWise SUB|Not supported in HW|Not supported in HW|
||EltWise MIN|&#10004;|Not implemented in SW|
||EltWise MAX|&#10004;|Not implemented in SW|
|**LRN**||&#10004;|Not implemented in SW|

### Frameworks support

|Framework &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;|Status &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;|
|---------|-------|
|Caffe|&#10004;|
|ONNX|Future|

### Networks verification report
 
|Network &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;|Configuration &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;|fp16 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |int8 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
|-------|----|----|----|
|MNIST|nv_full,nv_large,nv_small|Verified|Verified|
|ResNet-18|nv_full,nv_large,nv_small|Verified|Verified|
|ResNet-50|nv_full,nv_large,nv_small|Verified|Verified|

### Known limitations
- Not supported in HW
    - Dilation with Winograd
    - EltWise SUB
    - Pooling and convolution layers where pad size is greater than kernel size
- Not implemented in SW
    - Deconvolution with strides > 32
    - Deconvolution with input/output padding
 

