- Run 
```js
sudo docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/home/phucphan/freelancer/triton_server/model_repository:/models nvcr.io/nvidia/tritonserver:22.01-py3 tritonserver --model-repository=/models

python face_client.py -m ir50_onnx imgs/phucpx1.jpg


```