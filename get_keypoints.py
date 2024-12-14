import cv2
import time
import torch
import argparse
import numpy as np
import os
import json  # JSON 모듈 추가
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt, strip_optimizer, xyxy2xywh
from utils.plots import output_to_keypoint, plot_one_box_kpt, colors

def get_keypoints(image_path, poseweights="yolov7-w6-pose.pt", device='cpu'):
    device = select_device(device)  # 장치 선택
    model = attempt_load(poseweights, map_location=device)  # 모델 로드
    model.eval()

    # 이미지 읽기
    orig_image = cv2.imread(image_path)
    if orig_image is None:
        print('Error while trying to read image. Please check path again')
        return None

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)  # RGB로 변환
    image = letterbox(image, (640, 640), stride=64, auto=True)[0]
    image = transforms.ToTensor()(image).unsqueeze(0)  # 배치 추가
    image = image.to(device).float()  # 장치로 이동 및 float형 변환

    with torch.no_grad():  # 예측 진행
        output_data, _ = model(image)

    output_data = non_max_suppression_kpt(output_data, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)

    keypoints_data = []
    body_parts = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 
                  'LShoulder', 'LElbow', 'LWrist', 'RHip', 'RKnee', 
                  'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 
                  'LEye', 'REar', 'LEar']

    for pose in output_data:  # 이미지당 검출된 포즈 처리
        if pose is not None:
            for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:, :6])):
                kpts = pose[det_index, 6:]

                # 키포인트 데이터 저장
                keypoints = []
                for j in range(len(kpts) // 3):
                    if j < len(body_parts):
                        x_coord = kpts[j * 3]  # x 좌표
                        y_coord = kpts[j * 3 + 1]  # y 좌표
                        confidence = kpts[j * 3 + 2]  # 신뢰도

                        keypoints.append({
                            'body_part': body_parts[j],
                            'x': x_coord.item(),
                            'y': y_coord.item(),
                            'confidence': confidence.item()
                        })
                
                keypoints_data.append({
                    'object_index': det_index + 1,
                    'keypoints': keypoints
                })

    json_data = {
        "image_info": {
            "width": orig_image.shape[1],
            "height": orig_image.shape[0],
        },
        "annotations": keypoints_data,  # annotations에 keypoints_data 추가
    }
                

    return json_data  # 키포인트 배열 반환