![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/97fb822e-b52f-496f-9213-da9a8b169502/450fb0a5-42b1-46fa-a99d-2645061b220c/Untitled.png)

## 📌 **구성원 및 역할**

---

| 박수아 | 발표, 모델 설계 | VGG 19 설계, 발표 |
| --- | --- | --- |
| 심현지 | 모델 설계 | EDA, ResNet 34 |
| 이인철 | 모델 설계 | MobileNet, 시각화 |
| 이효준 | 모델 설계 | Cuda, ResNet 18 |
| 조용재 | 모델 설계 | EDA, ResNet 18 |

## 📌 **결과 보고서**

---

[CRNN 프로젝트 결과 보고서.pdf](https://prod-files-secure.s3.us-west-2.amazonaws.com/97fb822e-b52f-496f-9213-da9a8b169502/09bc434c-7e89-4a21-9a2c-3625a626a275/CRNN_%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8_%EA%B2%B0%EA%B3%BC_%EB%B3%B4%EA%B3%A0%EC%84%9C.pdf)

## 📌 개요

---

> DACON에서 진행했던 지난 2023 교원그룹 AI OCR 챌린지에 참가하며 본 프로젝트를 진행하게 되었다. AI가 학습에 적극적으로 활용되는 교육 시장의 흐름을 선도하고자 손글씨 인식에 최적화된 인공지능을 개발하고자 하는 취지에 맞게 다양한 cnn 모델을 사용하며 Text Recognition을 수행하는 인식 AI 모델을 개발하고자 했다. 다양한 교육 업계에서 ‘OCR’이 도입되고 있는 만큼 이러한 경험은 천재교과서 밀크티초등, 스마트 학습지 ‘필기 인식’ 기술 강화에 큰 도움이 될 수 있을 것으로 기대된다.
> 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/97fb822e-b52f-496f-9213-da9a8b169502/69a6eb11-82f6-48a6-9093-3bbd13a02993/Untitled.png)

## 📌 요약

---

> train_data : 폰트 손 글씨 학습 데이터, 76888개의 이미지
test_data : 폰트 손 글씨 평가 데이터, 74121개의 이미지
train.csv : id(샘플 고유 id), img_path(샘플 이미지 파일 경로), label(샘플 이미지에 해당하는 Text
test.csv : id(샘플 고유 id), img_path(샘플 이미지 파일 경로)

`OpenCV2를 이용해서 격자 탐색을 통해 이진화 처리를 진행`하여 train_data의 노이즈를 최소화했다. 메모리 한계로 인해 각 이미지를 gray scaling을 진행한 후 `Resnet 18, Resnet 34에 각각 추가적인 layer를 구성한 뒤 rnn 모델에 연결`하여 최종 모델 후보를 선정했다. vgg 19 역시 메모리 한계로 인해 최종 모델 후보에 선정되지 못했다.

최종 모델은 `Resnet34에 추가 layer를 구성한 뒤, rnn 모델에 연결한 모델`이 선정되었으며 직접 작성한 손 글씨를 바탕으로 성능을 확인했다.
> 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/97fb822e-b52f-496f-9213-da9a8b169502/fea9e7a6-6522-4c5e-948d-0572adcc8f91/Untitled.png)

## 📌 과정

---

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/97fb822e-b52f-496f-9213-da9a8b169502/ad7434bd-4b66-43f1-bc10-b9abd146d588/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/97fb822e-b52f-496f-9213-da9a8b169502/9ac43645-8596-424a-b89f-8f6670cd6130/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/97fb822e-b52f-496f-9213-da9a8b169502/1efd726f-9c56-434b-90fe-0bee3a60f679/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/97fb822e-b52f-496f-9213-da9a8b169502/0d2f9704-4fdf-497c-b128-9ab81ac258c6/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/97fb822e-b52f-496f-9213-da9a8b169502/2eca1ba8-7dd9-4ff8-8b7f-1522808c8223/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/97fb822e-b52f-496f-9213-da9a8b169502/05184f47-e2a8-4fd1-8052-cc13f8026faa/Untitled.png)

## 📌 결과

---

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/97fb822e-b52f-496f-9213-da9a8b169502/e83dc193-31aa-4c73-b565-a6787455c25a/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/97fb822e-b52f-496f-9213-da9a8b169502/204e8ee0-00b6-48e8-a22c-e49cd7f98617/Untitled.png)

## 📌 분석 활용 전략

---

### ✅ 밀크T 손 글씨 인식 기술 제안

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/97fb822e-b52f-496f-9213-da9a8b169502/408ec499-089d-40ca-995d-24a88554b11c/Untitled.png)

## 📌 개발 환경 / 툴

---

- VSCode
- Python
- OpenCV
- Pytorch
- Matplotlib
- Pillow
- Scikit-learn

## 📌 깃허브 주소

---
