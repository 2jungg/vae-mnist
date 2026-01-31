# VAE-MNIST 프로젝트

PyTorch로 구현한 Variational Autoencoder (VAE)를 MNIST 데이터셋으로 학습하고 시각화합니다.

## 환경 설정

```bash
# Conda 환경 활성화
conda activate vae-mnist

# 필요한 패키지 (이미 설치됨)
# pip install torch torchvision matplotlib numpy scikit-learn
```

## 프로젝트 구조

```
vae-mnist/
├── vae.py           # VAE 모델 정의
├── train.py         # 학습 스크립트
├── visualize.py     # 시각화 스크립트
├── models/          # 저장된 모델
├── results/         # 시각화 결과
└── data/            # MNIST 데이터셋 (자동 다운로드)
```

## 실행 방법

### 1. 학습
```bash
python train.py
```

### 2. 시각화
```bash
pip install scikit-learn  # t-SNE용
python visualize.py
```

## 모델 아키텍처

```
Input (784) → FC(400) → ReLU → [μ, logσ²] (20) → FC(400) → ReLU → Output (784)
```

- **Encoder**: 784 → 400 → 20 (latent)
- **Decoder**: 20 → 400 → 784
- **Latent dim**: 20
- **Loss**: BCE + KL Divergence

## 결과

| 항목 | 파일 |
|------|------|
| 학습 곡선 | `results/training_curve.png` |
| 재구성 비교 | `results/reconstruction.png` |
| Latent Space (2D) | `results/latent_space_2d.png` |
| Latent Space (t-SNE) | `results/latent_space_tsne.png` |
| 생성 샘플 | `results/generated_samples.png` |
| 보간 | `results/interpolation.png` |

## 학습 진행 상황

- [x] 환경 설정 완료
- [x] VAE 구현 완료
- [x] 학습 진행 완료
- [x] 시각화 완료
