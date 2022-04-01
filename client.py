# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan


import cv2
import json
import uuid
import random
import numpy as np
import tritonclient.http as httpclient
import tritonclient.utils.shared_memory as shm

from typing import List
from tritonclient.utils import *


CONFIDENCE_THRESOLD = 0.6


def plot_one_box(x, landmark, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
    param:
        x:     a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
            line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness

    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    cv2.circle(img, (int(landmark[0]), int(landmark[1])), 1, (0, 0, 255), 4) # eye_right
    cv2.circle(img, (int(landmark[2]), int(landmark[3])), 1, (0, 255, 255), 4) # eye_left
    cv2.circle(img, (int(landmark[4]), int(landmark[5])), 1, (255, 0, 255), 4) # nose
    cv2.circle(img, (int(landmark[6]), int(landmark[7])), 1, (0, 255, 0), 4)    # mouth_right
    cv2.circle(img, (int(landmark[8]), int(landmark[9])), 1, (255, 0, 0), 4) # mouth_left

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


class RegisterClient:
    def __init__(self, host="localhost:8000", model_name="register"):
        self.client = httpclient.InferenceServerClient(host)
        self.model_name = model_name

    def infer_by_path(self, frames, draw=False):
        frames = [cv2.imread(path, cv2.IMREAD_COLOR)]
        return self.infer_by_frame(frames, draw=draw)

    def infer_by_frame(self, frames, draw=False):
        assert  len(frames) == 1
        images_data = np.asarray(frames)
        if len(images_data.shape) == 3:
            images_data = np.expand_dims(images_data, axis=0)

        inputs = [
            httpclient.InferInput('IMAGE', images_data.shape, np_to_triton_dtype(images_data.dtype))
        ]
        inputs[0].set_data_from_numpy(images_data)
        outputs = [
            httpclient.InferRequestedOutput('OUTPUT')
        ]

        response = self.client.infer(
            self.model_name,
            inputs,
            request_id=uuid.uuid4().hex[:6],
            outputs=outputs
        )
        response.get_response()
        ret = response.as_numpy("OUTPUT")[0][0]
        ret = json.loads(ret)
        plot = None

        if len(ret) == 0:
            return {}, plot

        if draw:
            plot = images_data[0].copy()

            if len(ret):
                bbox = ret[0]["bbox"]
                landmark = []
                for x in ret[0]["landms"]:
                    landmark.extend(x)
                score = ret[0]["score"]
                plot_one_box(
                    bbox,
                    landmark,
                    plot,
                    label="{:.3f}".format(score),
                )

                return ret[0], plot

        return ret[0], plot

    def shutdown(self):
        self.client.close()


class ExtractorClient:
    def __init__(
            self,
            host="localhost:8000",
            model_name='ext_unc',
            using_shm=True,
            max_batch_size=32,
            image_input_shape=[112, 112, 3]
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
            images_data_size = max_batch_size * fake_image_data.size *fake_image_data.itemsize

            self.shm_image_data_hanlde = shm.create_shared_memory_region(
                self.shm_name,
                self.shm_key,
                images_data_size
            )
            self.client.register_system_shared_memory(
                self.shm_name,
                self.shm_key,
                images_data_size
            )

    def infer_by_frames(self, frames: np.ndarray, faces: List[np.ndarray]):
        faces_data = np.concatenate(faces, axis=0).astype(np.float32)
        nboxes_data = np.asarray([face.shape[0] for face in faces], dtype=np.uint32)

        assert frames.shape[1] == self.image_input_shape[0]
        assert frames.shape[2] == self.image_input_shape[1]

        if self.using_shm:
            shm.set_shared_memory_region(self.shm_image_data_handle, [frames])

        inputs = [
            httpclient.InferInput("IMAGE", frames.shape,
                                  np_to_triton_dtype(frames.dtype)),
            httpclient.InferInput("FACE_INFO", faces_data.shape,
                                  np_to_triton_dtype(faces_data.dtype)),
            httpclient.InferInput("NBOXES", nboxes_data.shape,
                                  np_to_triton_dtype(nboxes_data.dtype)),
        ]

        images_data_size = frames.size * frames.itemsize
        if self.using_shm:
            inputs[0].set_shared_memory(self.shm_name, images_data_size)
        else:
            inputs[0].set_data_from_numpy(frames)

        inputs[1].set_data_from_numpy(faces_data)
        inputs[2].set_data_from_numpy(nboxes_data)

        outputs = [
            httpclient.InferRequestedOutput("OUTPUT"),
        ]

        response = self.client.infer(
            self.model_name,
            inputs,
            request_id=uuid.uuid4().hex[:6],
            outputs=outputs
        )

        response.get_response()
        rets = response.as_numpy("OUTPUT")

        ret_decode = []
        for ret in rets:
            ret_decode.append(json.loads(ret))

        return ret_decode

    def shutdown(self):
        if self.using_shm:
            shm.destroy_shared_memory_region(self.shm_image_data_handle)
            self.client.unregister_system_shared_memory(name=self.shm_name)

        self.client.close()
