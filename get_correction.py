import cv2
import time
import torch
import argparse
import numpy as np
import os
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt, strip_optimizer, xyxy2xywh
from utils.plots import output_to_keypoint, plot_one_box_kpt, colors
from pose_correction import pose_correction

def correction(image_path, predicted_class=-1, poseweights="yolov7-w6-pose.pt", device='cuda', hide_labels=False, hide_conf=True, line_thickness=3, output_img="./output"):

    #device = torch.device('cuda')
    device = select_device(device)  # 장치 선택
    model = attempt_load(poseweights, map_location=device)  # 모델 로드
    model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names  # 클래스 이름 가져오기

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

    im0 = image[0].permute(1, 2, 0) * 255  # Change format [b, c, h, w] to [h, w, c] for displaying the image.
    im0 = im0.cpu().numpy().astype(np.uint8)
    
    im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)  # reshape image format to (BGR)

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

                # 라벨 설정
                label = None if hide_labels else (names[int(cls)] if hide_conf else f'{names[int(cls)]} {conf:.2f}')
                plot_one_box_kpt(xyxy, im0, label=label, color=colors(int(cls), True), 
                                  line_thickness=line_thickness, kpt_label=True, kpts=kpts, steps=3, 
                                  orig_shape=im0.shape[:2])
                print(f'kpts : {kpts}')


    correction_text = pose_correction(keypoints_data, predicted_class)

    # correction 텍스트 추가
    cv2.putText(im0, correction_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.567, (0, 0, 255), 2)


    # 결과 이미지 저장
    base_name = os.path.splitext(os.path.basename(image_path))[0]  # 입력 이미지 이름에서 확장자 제거

    # output 폴더가 존재하지 않으면 생성
    if not os.path.exists(output_img):
        os.makedirs(output_img)

    output_image_name = os.path.join(output_img, f"{base_name}.jpg")  # 저장 경로 및 이름 설정
    cv2.imwrite(output_image_name, im0)  # 결과 이미지 저장
    print(f"Output image saved as {output_image_name}")
