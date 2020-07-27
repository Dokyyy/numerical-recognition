#   Copyright (c) 2016, Xilinx, Inc.
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   1.  Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#   2.  Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#   3.  Neither the name of the copyright holder nor the names of its
#       contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#   THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#   OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
#   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
#   OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
#   ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pynq import Overlay, PL
from PIL import Image
import numpy as np
import cffi
import os
import tempfile

RUNTIME_HW = "python_hw"
RUNTIME_SW = "python_sw"

NETWORK_CNVW1A1 = "cnvW1A1"
NETWORK_CNVW1A2 = "cnvW1A2"
NETWORK_CNVW2A2 = "cnvW2A2"
NETWORK_LFCW1A1 = "lfcW1A1"
NETWORK_LFCW1A2 = "lfcW1A2"

if os.environ['BOARD'] == 'Ultra96':
    PLATFORM = "ultra96"
elif os.environ['BOARD'] == 'Pynq-Z1' or os.environ['BOARD'] == 'Pynq-Z2':
    PLATFORM = "pynqZ1-Z2"
else:
    raise RuntimeError("Board not supported")

BNN_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
BNN_LIB_DIR = os.path.join(BNN_ROOT_DIR, 'libraries', PLATFORM)
BNN_BIT_DIR = os.path.join(BNN_ROOT_DIR, 'bitstreams', PLATFORM)
BNN_PARAM_DIR = os.path.join(BNN_ROOT_DIR, 'params')

_ffi = cffi.FFI()

_ffi.cdef("""
void load_parameters(const char* path);
int inference(const char* path, int results[64], int number_class, float *usecPerImage);
int* inference_multiple(const char* path, int number_class, int *image_number, float *usecPerImage, int enable_detail);
void free_results(int * result);
void deinit();
"""
          )

_libraries = {}


# 检查数据集
def available_params(network):
    datasets = os.listdir(BNN_PARAM_DIR)
    ret = []
    for d in datasets:
        if os.path.isdir(os.path.join(BNN_PARAM_DIR, d)):
            supportedNets = os.listdir(os.path.join(BNN_PARAM_DIR, d))
            for nets in supportedNets:
                if nets == network:
                    ret.append(d)
    return ret


# pyhton对象作为接口，通过c++共享对象库与主机库通信
class PynqBNN:
    # 在创建PynqBNN时，加载给定网络的共享库并将比特流下载到PL
    # 当打算使用硬件加速运行时
    def __init__(self, runtime, network, load_overlay=True):
        self.bitstream_name = None
        if runtime == RUNTIME_HW:
            self.bitstream_name = "{0}-{1}.bit".format(network, PLATFORM)
            self.bitstream_path = os.path.join(BNN_BIT_DIR, self.bitstream_name)
            if PL.bitfile_name != self.bitstream_path:
                if load_overlay:
                    Overlay(self.bitstream_path).download()
                else:
                    raise RuntimeError("Incorrect Overlay loaded")
        dllname = "{0}-{1}-{2}.so".format(runtime, network, PLATFORM)
        if dllname not in _libraries:
            _libraries[dllname] = _ffi.dlopen(os.path.join(BNN_LIB_DIR, dllname))
        self.interface = _libraries[dllname]
        self.num_classes = 0

    def __del__(self):
        self.interface.deinit()

    # 函数设置特定网络的权重和激活阈值
    def load_parameters(self, params):
        if not os.path.isabs(params):
            params = os.path.join(BNN_PARAM_DIR, params)
        if os.path.isdir(params):
            self.interface.load_parameters(params.encode())
            self.classes = []
            with open(os.path.join(params, "classes.txt")) as f:
                self.classes = [c.strip() for c in f.readlines()]
            filter(None, self.classes)
        else:
            print("\nERROR: No such parameter directory \"" + params + "\"")

    # 开始对具有最高等级输出的单个图像进行推理
    def inference(self, path):
        usecperimage = _ffi.new("float *")
        result_ptr = self.interface.inference(path.encode(), _ffi.NULL, len(self.classes), usecperimage)
        print("Inference took %.2f microseconds" % (usecperimage[0]))
        print("Classification rate: %.2f images per second" % (1000000.0 / usecperimage[0]))
        self.usecPerImage = usecperimage[0]
        return result_ptr


    # 在单个图像上开始推理，输出包含所有可用类簇的排名的向量
    def detailed_inference(self, path):
        details_ptr = _ffi.new("int[]", len(self.classes))
        usecperimage = _ffi.new("float *")
        self.interface.inference(path.encode(), details_ptr, len(self.classes), usecperimage)
        details_buf = _ffi.buffer(details_ptr, len(self.classes) * 4)
        print("Inference took %.2f microseconds" % (usecperimage[0]))
        print("Classification rate: %.2f images per second" % (1000000.0 / usecperimage[0]))
        details_array = np.copy(np.frombuffer(details_buf, dtype=np.int32))
        self.usecPerImage = usecperimage[0]
        return details_array

    # 开始对多个图像进行推理，输出是包含每个图像的推理类簇的向量
    def inference_multiple(self, path):
        size_ptr = _ffi.new("int *")
        usecperimage = _ffi.new("float *")
        result_ptr = self.interface.inference_multiple(path.encode(), len(self.classes), size_ptr, usecperimage, 0)
        result_buffer = _ffi.buffer(result_ptr, size_ptr[0] * 4)
        print(
            "Inference took %.2f microseconds, %.2f usec per image" % (usecperimage[0] * size_ptr[0], usecperimage[0]))
        result_array = np.copy(np.frombuffer(result_buffer, dtype=np.int32))
        print("Classification rate: %.2f images per second" % (1000000.0 / usecperimage[0]))
        self.interface.free_results(result_ptr)
        self.usecPerImage = usecperimage[0]
        return result_array

    # 开始对多个图像进行推理，输出包含每个类对每个图像的排名
    def inference_multiple_detail(self, path):
        size_ptr = _ffi.new("int *")
        usecperimage = _ffi.new("float *")
        result_ptr = self.interface.inference_multiple(path.encode(), len(self.classes), size_ptr, usecperimage, 1)
        print(
            "Inference took %.2f microseconds, %.2f usec per image" % (usecperimage[0] * size_ptr[0], usecperimage[0]))
        print("Classification rate: %.2f images per second" % (1000000.0 / usecperimage[0]))
        result_buffer = _ffi.buffer(result_ptr, len(self.classes) * size_ptr[0] * 4)
        result_array = np.copy(np.frombuffer(result_buffer, dtype=np.int32))
        self.interface.free_results(result_ptr)
        self.usecPerImage = usecperimage[0]
        return result_array

    def class_name(self, index):
        return self.classes[index]


# CNV网络的分类器class，用于对cifar10格式的图像或需要预处理的图像进行推理
class CnvClassifier:

    # 构造函数将加载共享库，将比特流下载到PL，并将参数集加载到网络中
    def __init__(self, network, params, runtime=RUNTIME_HW):
        if params in available_params(network):
            self.net = network
            self.params = params
            self.runtime = runtime
            self.usecPerImage = 0.0
            self.bnn = PynqBNN(runtime, network) # 创建PynqBNN时，加载给定网络的共享库并将比特流下载到PL
            self.bnn.load_parameters(os.path.join(params, network)) # 加载参数到网络
            self.classes = self.bnn.classes
        else:
            print("ERROR: parameters are not availlable for {0}".format(network))

    # 将图像转换为cifar10格式
    def image_to_cifar(self, img, fp):
        # 将下载的图像调整为32x32像素以便BNN处理
        img.thumbnail((32, 32), Image.ANTIALIAS)
        background = Image.new('RGBA', (32, 32), (255, 255, 255, 0))
        background.paste(
            img, (int((32 - img.size[0]) / 2), int((32 - img.size[1]) / 2))
        )

        img = (np.array(background))
        r = img[:, :, 0].flatten()
        g = img[:, :, 1].flatten()
        b = img[:, :, 2].flatten()
        label = np.identity(1, dtype=np.uint8)
        fp.write(label.tobytes())
        fp.write(r.tobytes())
        fp.write(g.tobytes())
        fp.write(b.tobytes())

    # 对非cifar10格式的图像进行分类，结果为最可能的分类
    def classify_image(self, img):
        with tempfile.NamedTemporaryFile() as tmp:
            self.image_to_cifar(img, tmp)
            tmp.flush()
            result = self.bnn.inference(tmp.name)
        self.usecPerImage = self.bnn.usecPerImage
        return result

    # 对cifar10格式的图像进行分类
    def classify_cifar(self, path):
        result = self.bnn.inference(path)
        self.usecPerImage = self.bnn.usecPerImage
        return result

    # 对非cifar10格式的图像进行分类，结果为各类的所有排名向量
    def classify_image_details(self, img):
        with tempfile.NamedTemporaryFile() as tmp:
            self.image_to_cifar(img, tmp)
            tmp.flush()
            result = self.bnn.detailed_inference(tmp.name)
        self.usecPerImage = self.bnn.usecPerImage
        return result

    # 对cifar10格式的图像进行分类，结果为各类的所有排名向量
    def classify_cifar_details(self, path):
        result = self.bnn.detailed_inference(path)
        self.usecPerImage = self.bnn.usecPerImage
        return result

    # 对路径中的图像进行分类(仅对常规图像)
    def classify_path(self, path):
        img = Image.open(path)
        return self.classify_image(img)

    def classify_images(self, imgs):
        with tempfile.NamedTemporaryFile() as tmp:
            for img in imgs:
                self.image_to_cifar(img, tmp)
            tmp.flush()
            result = self.bnn.inference_multiple(tmp.name)
        self.usecPerImage = self.bnn.usecPerImage
        return result

    # 分类多个cifar10预格式化图片，输出推断类
    def classify_cifars(self, path):
        result = self.bnn.inference_multiple(path)
        self.usecPerImage = self.bnn.usecPerImage
        return result

    def classify_images_details(self, imgs):
        with tempfile.NamedTemporaryFile() as tmp:
            for img in imgs:
                self.image_to_cifar(img, tmp)
            tmp.flush()
            result = self.bnn.inference_multiple_detail(tmp.name)
        self.usecPerImage = self.bnn.usecPerImage
        return result

    def classify_cifars_details(self, path):
        result = self.bnn.inference_multiple_detail(path)
        self.usecPerImage = self.bnn.usecPerImage
        return result

    def classify_paths(self, paths):
        return self.classify_images([Image.open(p) for p in paths])

    def class_name(self, index):
        return self.bnn.classes[index]


# 分类器class，用于LFC网络对mnist格式的图像进行推理识别
class LfcClassifier:

    def __init__(self, network, params, runtime=RUNTIME_HW):
        if params in available_params(network):
            self.net = network
            self.params = params
            self.runtime = runtime
            self.usecPerImage = 0.0
            self.bnn = PynqBNN(runtime, network)
            self.bnn.load_parameters(os.path.join(params, network))
            print(os.path.join(params, network))
            self.classes = self.bnn.classes
        else:
            print("ERROR: parameters are not availlable for {0}".format(network))

    def classify_mnist(self, mnist_format_file):
        result = self.bnn.inference(mnist_format_file)
        self.usecPerImage = self.bnn.usecPerImage
        return result

    def classify_mnists(self, mnist_format_file):
        result = self.bnn.inference_multiple(mnist_format_file)
        self.usecPerImage = self.bnn.usecPerImage
        return result

    def class_name(self, index):
        return self.bnn.classes[index]