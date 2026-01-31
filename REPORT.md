# VAE-MNIST 실험 프로젝트 결과 리포트

## 1. 개요
본 프로젝트는 PyTorch를 사용하여 Variational Autoencoder(VAE)를 구현하고 MNIST 데이터셋을 통해 이미지 재구성 및 잠재 공간(Latent Space) 시각화 실험을 진행하였습니다.

## 2. 실험 환경
- **언어**: Python 3.10
- **프레임워크**: PyTorch
- **데이터셋**: MNIST (60,000 학습, 10,000 테스트)
- **장치**: CPU (사용 가능한 경우 GPU 자동 선택)

## 3. 모델 아키텍처
- **Encoder**: 784 (Input) -> 400 (Hidden) -> 20 (Latent Mu/LogVar)
- **Decoder**: 20 (Latent) -> 400 (Hidden) -> 784 (Output)
- **잠재 차원(Latent Dim)**: 20

## 4. 학습 결과
- **에폭(Epochs)**: 20
- **최종 학습 손실(Avg Loss)**: 103.8146
- **최종 테스트 손실(Avg Loss)**: 103.7815
- **특이사항**: 학습이 안정적으로 진행되었으며, 손실값이 지속적으로 감소하였습니다.

## 5. 시각화 분석
- **재구성 이미지(`results/reconstruction.png`)**: 원본 이미지의 특징을 잘 포착하여 선명하게 재구성함.
- **잠재 공간 시각화(`results/latent_space_tsne.png`)**: t-SNE를 통해 20차원 잠재 공간을 2차원으로 투영한 결과, 숫자별로 클러스터링이 잘 이루어짐.
- **이미지 생성(`results/generated_samples.png`)**: 잠재 공간에서 무작위 샘플링을 통해 새로운 숫자 이미지를 생성함.
- **보간(`results/interpolation.png`)**: 두 숫자 사이의 잠재 공간 보간을 통해 형태가 부드럽게 변하는 것을 확인.

## 6. 결론
VAE 모델이 MNIST 숫자의 유의미한 특징을 성공적으로 학습하였으며, 이를 통해 이미지 재구성 및 생성이 가능함을 확인하였습니다. 모든 결과물은 `/home/ubuntu/clawd/projects/vae-mnist/` 디렉토리에 저장되었습니다.
