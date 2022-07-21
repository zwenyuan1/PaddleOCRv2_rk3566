# PaddleOCRv2_rk3566

## 3rdparty
1.Download 3rdparty from [rknup2](https://github.com/rockchip-linux/rknpu2), then make dirs structure like:

- 3rdparty
- rknn_ocr_demo
     - include
     - src
     - install
     - CMakeLists.txt
     - build-linux_RK356X.sh
 
 2.Download cross-compile tool [host](), then modify the GCC_COMPILER and TOOL_CHAIN in build-linux_RK356X.sh 
 
## How to run
```
./build-linux_RK356X.sh
adb push ./install/rknn_ocr_demo_Linux /data
adb shell
cd /data/rknn_ocr_demo_Linux
./rknn_ocr_demo ./model/test_imgs/turn
```
### My CSDN about this: [rk3566部署paddleocrv2踩坑记录](https://blog.csdn.net/Emery_learning/article/details/125739825)
