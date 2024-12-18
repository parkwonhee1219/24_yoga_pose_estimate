import os
import torch
import torch.nn as nn
import argparse
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
from myNetLight import myNetLight 
from get_skeleton import data_keypoints
from get_keypoints import get_keypoints
from get_correction import correction
import time 

# 디바이스 설정 (GPU가 있으면 GPU 사용)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로드
def load_model(model_path, num_classes=3):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = myNetLight(num_classes=num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device,  weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 평가 모드로 전환
    return model

# 이미지 전처리
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # 이미지 읽기
    image = Image.open(image_path).convert('RGB')
    
    # PyTorch 텐서로 변환 및 정규화
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Imagenet 평균, 표준편차
    ])
    
    image = transform(image).unsqueeze(0).to(device)  # 배치 차원 추가
    return image

# 추론 함수
def infer(model, image, skeleton):

    start_time = time.time()
    with torch.no_grad():  # 기울기 계산 안 함
        outputs = model(image, skeleton)  # 모델의 출력 값
        _, predicted = torch.max(outputs, 1)  # 가장 높은 값을 가진 클래스 예측
    end_time = time.time()
    start_to_end = end_time - start_time
    print(f'time : {start_to_end}')
    return predicted.item()

# Command line argument 파싱
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./data_image/plank1.jpg', help='Image source path')
    parser.add_argument('--model', type=str, default='./resnet_light.pth', help='Path to the trained model')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to run the model on')
    return parser.parse_args()

# main 함수
def main():
    # 사용자 입력 처리
    opt = parse_opt()
    
    # 모델 로드
    model = load_model(opt.model)

    # 이미지 및 스켈레톤 전처리
    image = preprocess_image(opt.source)  # 입력 이미지 경로
    skeleton = data_keypoints(opt.source, "yolov7-w6-pose.pt", device)

    # 추론 수행
    predicted_class = infer(model, image, skeleton)

    # 예측된 클래스 출력
    print(f"Predicted class: {predicted_class}")

    # 자세 교정
    correction(opt.source, predicted_class)


if __name__ == "__main__":
    main()
