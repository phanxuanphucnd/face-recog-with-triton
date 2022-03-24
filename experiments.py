# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

import cv2
import onnx
import json
import torch
import timeit
import numpy as np
import onnxruntime
from model_irse import IR_50


def get_feats_from_onnx(
        img_path='./data/phucpx1.jpg',
        model_path = './models/ir50/ir50_onnx/backbone_ir50_asia.onnx'
):
    print(f"\nGet features from onnx")
    if torch.cuda.is_available():
        net = onnx.load(model_path)

    print(f"onnx device: {onnxruntime.get_device()}")

    providers = [
        ('TensorrtExecutionProvider', {
            'device_id': 0,
            'trt_max_workspace_size': 2147483648,
            'trt_fp16_enable': True,
        }),
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        })
    ]

    if torch.cuda.is_available():
        # session = onnxruntime.InferenceSession(model_path, None, providers=["CUDAExecutionProvider"])
        session = onnxruntime.InferenceSession(model_path, None, providers=providers)
    else:
        session = onnxruntime.InferenceSession(model_path, None)

    print(f"session providers: {session.get_providers()}")

    now = timeit.default_timer()
    INPUT_SIZE = [112, 112]

    img = cv2.imread(img_path)
    img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    data = json.dumps({'data': img.tolist()})
    data = np.array(json.loads(data)['data']).astype('float32')

    # net = IR_50(INPUT_SIZE)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: data})
    time = timeit.default_timer() - now

    return outputs, time


def get_feats_from_pth(
        img_path='./data/phucpx1.jpg',
        model_path = './models/ir50/backbone_ir50_asia.pth'
):
    now = timeit.default_timer()
    INPUT_SIZE = [112, 112]

    img = cv2.imread(img_path)
    img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    # net = get_model(name, fp16=False)
    net = IR_50(INPUT_SIZE)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
    else:
        net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    net.eval()
    # Inference
    print('--> Running model')
    outputs = net(img).detach().numpy()

    time = timeit.default_timer() - now

    return outputs, time


if __name__ == '__main__':
    # feat1 = get_feats_from_rknn(img_path='./data/phucpx1.jpg')
    feat2, time = get_feats_from_pth(img_path='./imgs/phucpx1.jpg', model_path='./models/ir50/backbone_ir50_asia.pth')
    print(type(feat2), np.shape(feat2))
    print(f"Time inference with .pt: {time}")
    feat3, time = get_feats_from_onnx(img_path='./imgs/phucpx1.jpg', model_path='./models/ir50_onnx/backbone_ir50_asia.onnx')
    print(type(feat3), np.shape(feat3))
    print(f"Time inference with .onnx: {time}")
    cosin = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    output = cosin(torch.Tensor(feat3[0]), torch.Tensor(feat2))

    print(f"Similarity = {output}")