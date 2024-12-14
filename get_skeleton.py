import cv2
import torch
import numpy as np
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt, strip_optimizer, xyxy2xywh

def data_keypoints(image_path, poseweights="yolov7-w6-pose.pt", device='cpu'):
    device = str(device)  # 'cuda' 또는 'cpu'로 변환
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

    body_parts = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 
                  'LShoulder', 'LElbow', 'LWrist', 'RHip', 'RKnee', 
                  'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 
                  'LEye', 'REar', 'LEar']

    keypoints = np.zeros((17,2), dtype=np.float32)

    for pose in output_data:  # 이미지당 검출된 포즈 처리
        if pose is not None:
            for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:, :6])):
                kpts = pose[det_index, 6:]

                # 키포인트 데이터 저장
                for j in range(len(kpts) // 3):
                    if j < len(body_parts):
                        x_coord = kpts[j * 3]  # x 좌표
                        y_coord = kpts[j * 3 + 1]  # y 좌표
                        confidence = kpts[j * 3 + 2]  # 신뢰도

                        keypoints[j] = [x_coord.item(), y_coord.item()]

    skeleton_tensor = torch.tensor(keypoints, dtype=torch.float32).to(device)
    skeleton_tensor = skeleton_tensor.unsqueeze(0) #배치 차원 추가

    return skeleton_tensor