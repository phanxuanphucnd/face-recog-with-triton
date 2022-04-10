# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan


import cv2
import json
import uuid
import random
import timeit
import numpy as np
import tritonclient.http as httpclient
from torchvision import transforms as trans
import tritonclient.utils.shared_memory as shm

from typing import List
from tritonclient.utils import *


class RecognitionClient:
    def __init__(
            self,
            host="localhost:8000",
            model_name='ir50_onnx',
            using_shm=True,
            max_batch_size=32,
            image_input_shape=[3, 112, 112]
    ):
        self.client = httpclient.InferenceServerClient(host)
        self.model_name = model_name
        self.using_shm = using_shm
        self.max_batch_size = max_batch_size
        self.image_input_shape = image_input_shape

        if self.using_shm:
            self.shm_name = f"image_data_{uuid.uuid4().hex[:6]}"
            self.shm_key = "0"

            fake_image_data = np.zeros([self.image_input_shape[0], self.image_input_shape[1], 3], dtype=np.uint8)
            images_data_size = max_batch_size * fake_image_data.size * fake_image_data.itemsize

            self.shm_image_data_handle = shm.create_shared_memory_region(
                self.shm_name,
                self.shm_key,
                images_data_size
            )
            self.client.register_system_shared_memory(
                self.shm_name,
                self.shm_key,
                images_data_size
            )

    def infer_by_frames(
            self,
            image_data: np.ndarray,
            input_name: str='input.1',
            output_name: str='559'
    ):
        assert image_data.shape[1] == self.image_input_shape[0]
        assert image_data.shape[2] == self.image_input_shape[1]

        if self.using_shm:
            shm.set_shared_memory_region(self.shm_image_data_handle, [frames])

        inputs = [
            httpclient.InferInput(input_name, image_data.shape, np_to_triton_dtype(image_data.dtype))
        ]


        images_data_size = image_data.size * image_data.itemsize
        if self.using_shm:
            inputs[0].set_shared_memory(self.shm_name, images_data_size)
        else:
            inputs[0].set_data_from_numpy(image_data)

        outputs = [
            httpclient.InferRequestedOutput(output_name)
        ]

        now = timeit.default_timer()
        response = self.client.infer(
            self.model_name,
            inputs,
            request_id=uuid.uuid4().hex[:6],
            outputs=outputs
        )
        print(">>> ", timeit.default_timer() - now)

        response.get_response()
        rets = response.as_numpy(output_name)

        return rets

    def shutdown(self):
        if self.using_shm:
            shm.destroy_shared_memory_region(self.shm_image_data_handle)
            self.client.unregister_system_shared_memory(name=self.shm_name)

        self.client.close()


def preprocess(img):
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = aug(img)
    img = transforms(img).data.cpu().numpy().astype(np.float32)

    return img

def compute(img, min_percentile, max_percentile):
    """Calculate the quantile, the purpose is to remove the abnormal situation at both ends of the histogram in Figure 1 """
    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)

    return max_percentile_pixel, min_percentile_pixel


def aug(src):
    """Image brightness enhancement"""
    max_percentile_pixel, min_percentile_pixel = compute(src, 1, 99)

    src[src >= max_percentile_pixel] = max_percentile_pixel
    src[src <= min_percentile_pixel] = min_percentile_pixel

    out = np.zeros(src.shape, src.dtype)
    cv2.normalize(src, out, 255 * 0.1, 255 * 0.9, cv2.NORM_MINMAX)
    return out


transforms = trans.Compose([
    trans.ToPILImage(),
    trans.Resize((112, 112)),
    trans.ToTensor(),
    trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

if __name__ == '__main__':

    imgs_path = [
        'imgs/thuyen1.jpeg',
        'imgs/thuyen2.jpeg',
        'imgs/thuyen3.jpeg',
        'imgs/thuyen4.jpeg',
        'imgs/thuyen4.jpeg',
        'imgs/thuyen4.jpeg',
        'imgs/thuyen4.jpeg'
    ]
    image_data = np.stack([preprocess(cv2.imread(x, cv2.IMREAD_COLOR)) for x in imgs_path])
    recognition_client = RecognitionClient(model_name='ir50_onnx', using_shm=False, max_batch_size=32)

    now = timeit.default_timer()

    decoded_output = recognition_client.infer_by_frames(image_data, input_name='input.1', output_name='559')

    print(timeit.default_timer() - now)


