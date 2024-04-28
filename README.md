![image](https://github.com/glassesholder/daycon_crnn_project/assets/150658909/d0519281-9496-4d58-8282-96929cbbb028)


## 📌 **구성원 및 역할**

---

| 박수아 | 발표, 모델 설계 | VGG 19 설계, 발표 |
| --- | --- | --- |
| 심현지 | 모델 설계 | EDA, ResNet 34 |
| 이인철 | 모델 설계 | MobileNet, 시각화 |
| 이효준 | 모델 설계 | Cuda, ResNet 18 |
| 조용재 | 모델 설계 | EDA, ResNet 18 |

## 📌 개요

---

> DACON에서 진행했던 지난 2023 교원그룹 AI OCR 챌린지에 참가하며 본 프로젝트를 진행하게 되었다. AI가 학습에 적극적으로 활용되는 교육 시장의 흐름을 선도하고자 손글씨 인식에 최적화된 인공지능을 개발하고자 하는 취지에 맞게 다양한 cnn 모델을 사용하며 Text Recognition을 수행하는 인식 AI 모델을 개발하고자 했다. 다양한 교육 업계에서 ‘OCR’이 도입되고 있는 만큼 이러한 경험은 천재교과서 밀크티초등, 스마트 학습지 ‘필기 인식’ 기술 강화에 큰 도움이 될 수 있을 것으로 기대된다.
> 

![image](https://github.com/glassesholder/daycon_crnn_project/assets/150658909/be5d2352-320d-4f1c-9ad2-0ad813b8b061)


## 📌 요약

---

> train_data : 폰트 손 글씨 학습 데이터, 76888개의 이미지
test_data : 폰트 손 글씨 평가 데이터, 74121개의 이미지
train.csv : id(샘플 고유 id), img_path(샘플 이미지 파일 경로), label(샘플 이미지에 해당하는 Text
test.csv : id(샘플 고유 id), img_path(샘플 이미지 파일 경로)

`OpenCV2를 이용해서 격자 탐색을 통해 이진화 처리를 진행`하여 train_data의 노이즈를 최소화했다. 메모리 한계로 인해 각 이미지를 gray scaling을 진행한 후 `Resnet 18, Resnet 34에 각각 추가적인 layer를 구성한 뒤 rnn 모델에 연결`하여 최종 모델 후보를 선정했다. vgg 19 역시 메모리 한계로 인해 최종 모델 후보에 선정되지 못했다.

최종 모델은 `Resnet34에 추가 layer를 구성한 뒤, rnn 모델에 연결한 모델`이 선정되었으며 직접 작성한 손 글씨를 바탕으로 성능을 확인했다.
> 

![image](https://github.com/glassesholder/daycon_crnn_project/assets/150658909/4b1ea858-11ae-43b0-824a-00d26b13181a)


## 📌 과정

---

![image](https://github.com/glassesholder/daycon_crnn_project/assets/150658909/e45bde26-f6bf-4954-95cd-e25873a4a11b)


![image](https://github.com/glassesholder/daycon_crnn_project/assets/150658909/17ca9f95-e0f4-450f-85eb-f4bcd734de42)


![image](https://github.com/glassesholder/daycon_crnn_project/assets/150658909/f38ad932-7f61-4f69-9650-de556c126ba1)


![image](https://github.com/glassesholder/daycon_crnn_project/assets/150658909/eab1f911-8ef8-4e3a-b80c-d42b1c9eeb67)


![image](https://github.com/glassesholder/daycon_crnn_project/assets/150658909/f031b268-8db3-44c4-a9ad-fe02922851f4)


![image](https://github.com/glassesholder/daycon_crnn_project/assets/150658909/adc8f2c6-c551-4022-92b6-23d1013c1f76)


## 📌 결과

---

![image](https://github.com/glassesholder/daycon_crnn_project/assets/150658909/fa40163d-db02-4f9c-9645-ad0a8029b8d6)


![image](https://github.com/glassesholder/daycon_crnn_project/assets/150658909/8db457ce-558c-4593-ad84-326c34f2b546)

## 📌 분석 활용 전략

---

### ✅ 밀크T 손 글씨 인식 기술 제안

![image](https://github.com/glassesholder/daycon_crnn_project/assets/150658909/9e434bba-0131-4341-b3a7-c56bb5946485)

## 📌 개발 환경 / 툴

---

- VSCode
- Python
- OpenCV
- Pytorch
- Matplotlib
- Pillow
- Scikit-learn
