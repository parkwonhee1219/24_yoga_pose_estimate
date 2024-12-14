import torch
import torch.nn as nn

# Lightweight ResNet Implementation
class ResidualBlock(nn.Module): #Residual Block 정의하는 클래스
    def __init__(self, in_channels, out_channels, strides): #클래스 초기화될 때 호출, 입력채널수, 출력채널수, 스트라이드를 인자로 받음.
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels), #배치 정규화 (각 배치의 평균을 0, 분산을 1로 조정)
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=1, stride=1, bias=False),  # Output channels x2
            nn.BatchNorm2d(out_channels * 2)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 2, kernel_size=1, stride=strides, bias=False),
            nn.BatchNorm2d(out_channels * 2)
        )
        self.relu = nn.ReLU()

    def forward(self, x): #자동호출
        residual = x
        x = self.conv_block(x)
        if x.shape != residual.shape:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x

class myNetLight(nn.Module):
    def __init__(self, num_classes=3):
        super(myNetLight, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.res_block1 = nn.Sequential(
            ResidualBlock(16, 16, strides=1),
            *[ResidualBlock(32, 16, strides=1) for _ in range(1)]  # Reduced number of blocks
        )
        self.res_block2 = nn.Sequential(
            ResidualBlock(32, 32, strides=2),
            *[ResidualBlock(64, 32, strides=1) for _ in range(2)]
        )
        self.res_block3 = nn.Sequential(
            ResidualBlock(64, 64, strides=2),
            *[ResidualBlock(128, 64, strides=1) for _ in range(2)]
        )
        self.res_block4 = nn.Sequential(
            ResidualBlock(128, 128, strides=2),
            *[ResidualBlock(256, 128, strides=1) for _ in range(1)]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # MLP for Skeleton Data 이건 MLP학습
        self.skeleton_mlp = nn.Sequential(
            nn.Linear(17 * 2, 128),  # 17 keypoints, each with x, y(17*2의 채널) #학습은 linear 레이어로 했다.
            nn.ReLU(), #활성화함수 넣어주면서 늘렸다 줄였다
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32) #최종 채널 32 (이 채널 값 32인 상태에서 레이어 수정해보기 / 32도 다른 수로 바꿀 수 있다)
        )

        # Combined Fully Connected Layer
        self.combined_fc = nn.Linear(256 + 32, num_classes)  # Adjusted input size to match reduced ResNet

    def forward(self, image, skeleton):  #2개의 input 1개의 output(0,1,2 중 하나)
        # Image branch
        x = self.conv1(image) #cnn
        x = self.res_block1(x) #여기가 resnet
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) #1차원으로 펼치기

        # Skeleton branch
        skeleton = torch.flatten(skeleton, 1)  # Flatten skeleton data
        skeleton_features = self.skeleton_mlp(skeleton)

        # Combine features
        combined = torch.cat((x, skeleton_features), dim=1) #concat계산하기
        out = self.combined_fc(combined) #3으로 줄이는거 호출 => 최종 output은 3개

        return out
