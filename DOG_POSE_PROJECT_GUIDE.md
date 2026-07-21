# AIHub 반려견 Pose Estimation 프로젝트 정리

이 문서는 AIHub 반려견 보행 데이터셋을 이용해 YOLO Pose 기반 keypoint detection 모델을 학습하기 위해 알아야 하는 핵심 구조, 데이터 변환 방식, 학습 전략, 주의사항을 정리한 문서다.

## 1. 프로젝트 목표

AIHub 반려견 보행 이미지/라벨 데이터를 이용하여 강아지의 주요 해부학적 관절점을 추정하는 YOLO Pose 모델을 학습한다.

목표는 단순 객체 검출이 아니라, 보행 분석에 활용 가능한 skeleton keypoint를 안정적으로 추정하는 것이다.

현재 구조는 view 특성에 따라 두 모델로 분리한다.

```text
front_back model:
  Front, Back 이미지 담당

side model:
  Left, Right 이미지 담당
```

전/후방과 측면은 keypoint 가시성 및 좌우 판단 방식이 다르므로, 하나의 모델로 통합하지 않고 2-model 구조를 사용한다.

---

## 2. 원본 AIHub 데이터 구조

원본 데이터셋은 대략 다음 구조를 가진다.

```text
G:\aihub_dog_pose
  train/
    image/
    label/
  validation/
    image/
    label/
```

하위 구조 예시는 다음과 같다.

```text
1기/
  2024_08_30_17_12_30/
    Back/
    Front/
    Left/
    Right/
```

중요한 점:

```text
원본 train/validation split은 그대로 신뢰하지 않는다.
같은 촬영 세션의 유사 프레임이 train과 validation에 섞일 수 있다.
따라서 session-level split을 새로 만든다.
```

---

## 3. 원본 JSON 라벨 구조

원본 라벨은 YOLO 형식이 아니라 JSON 형식이다.

예시:

```json
{
  "image_info": {
    "filename": "...",
    "resolution": "1920X1080"
  },
  "annotation_info": [
    {
      "x": "0.413",
      "y": "0.319",
      "label": "Iliac crest"
    }
  ]
}
```

특징:

```text
x, y는 0~1 normalized 좌표
visibility 값 없음
label 이름만 존재
좌/우 anatomical side 정보 없음
```

YOLO Pose 변환 시 visibility는 다음처럼 처리한다.

```text
라벨이 존재하는 keypoint → v = 2
라벨이 없는 keypoint → v = 0
```

---

## 4. Keypoint schema

원본 라벨 종류는 12종이다.

```text
Ear
T13 spinous process
Dorsal scapular spine
Acromion / Greater tubercle
Lateral humeral epicondyle
Ulnar styloid process
Distal lateral aspect of fifth metacarpal bone
Iliac crest
Femoral greater trochanter
Femorotibial joint
Lateral malleolus
Distal lateral aspect of fifth metatarsal
```

하지만 실제 데이터에서는 일부 landmark가 좌우 양쪽에 존재할 수 있다.

따라서 최종 모델은 22-keypoint 구조를 사용한다.

```text
T13 spinous process → 1개
Iliac crest → 1개
나머지 10개 landmark → left/right 2개씩

2 + 10 × 2 = 22 keypoints
```

최종 keypoint order:

```text
0  left_ear
1  right_ear
2  t13_spinous_process
3  left_dorsal_scapular_spine
4  left_acromion_greater_tubercle
5  left_lateral_humeral_epicondyle
6  left_ulnar_styloid_process
7  left_fifth_metacarpal_distal
8  right_dorsal_scapular_spine
9  right_acromion_greater_tubercle
10 right_lateral_humeral_epicondyle
11 right_ulnar_styloid_process
12 right_fifth_metacarpal_distal
13 iliac_crest
14 left_femoral_greater_trochanter
15 left_femorotibial_joint
16 left_lateral_malleolus
17 left_fifth_metatarsal_distal
18 right_femoral_greater_trochanter
19 right_femorotibial_joint
20 right_lateral_malleolus
21 right_fifth_metatarsal_distal
```

주의:

```text
이 순서는 절대 바꾸면 안 된다.
YOLO txt 라벨, yaml, 추론 skeleton, 시각화 코드가 모두 이 순서에 의존한다.
```

---

## 5. Flip index

YOLO Pose에서 horizontal flip augmentation을 사용하려면 `flip_idx`가 필요하다.

22-keypoint 기준 flip index:

```text
[1, 0, 2, 8, 9, 10, 11, 12, 3, 4, 5, 6, 7, 13, 18, 19, 20, 21, 14, 15, 16, 17]
```

의미:

```text
left_ear ↔ right_ear
left front limb ↔ right front limb
left hind limb ↔ right hind limb
T13, Iliac crest는 자기 자신
```

---

## 6. Skeleton 연결 구조

시각화용 skeleton은 다음 구조를 사용한다.

```text
(0, 2), (1, 2), (2, 13)
(2, 3), (3, 4), (4, 5), (5, 6), (6, 7)
(2, 8), (8, 9), (9, 10), (10, 11), (11, 12)
(13, 14), (14, 15), (15, 16), (16, 17)
(13, 18), (18, 19), (19, 20), (20, 21)
```

학습 자체에는 skeleton이 직접 필요하지 않지만, 추론 영상에서 keypoint line을 그릴 때 필요하다.

---

## 7. View별 좌우 매핑

### 7.1 Front / Back

전/후방 이미지는 x좌표를 이용해 좌우를 나눌 수 있다.

기준은 화면 기준 좌우가 아니라 강아지의 anatomical left/right다.

```text
Front:
  화면 오른쪽 → 강아지 left
  화면 왼쪽 → 강아지 right

Back:
  화면 왼쪽 → 강아지 left
  화면 오른쪽 → 강아지 right
```

한 라벨이 2개 있으면 x좌표를 기준으로 나눈다.

한 라벨만 있을 경우에는 몸 중심선 기준으로 좌우를 판단한다.

몸 중심선 기준:

```text
T13 spinous process
Iliac crest
```

### 7.2 Left / Right side

측면 이미지는 x좌표만으로 anatomical left/right를 판단하기 어렵다.

기존 방식:

```text
Left view  → 기본 라벨은 left keypoint
Right view → 기본 라벨은 right keypoint
```

개선 예정 방식:

```text
Left view:
  기본 라벨      → left keypoint
  Opposite 라벨 → right keypoint

Right view:
  기본 라벨      → right keypoint
  Opposite 라벨 → left keypoint
```

---

## 8. YOLO Pose 라벨 변환

YOLO Pose 라벨은 `.txt` 한 줄로 구성된다.

```text
class x_center y_center width height kpt1_x kpt1_y kpt1_v ... kpt22_x kpt22_y kpt22_v
```

class는 dog 하나만 있으므로 항상 0이다.

```text
class = 0
```

bbox는 visible keypoint를 감싸는 박스로 자동 생성한다.

```text
visible keypoints의 min/max x/y
+ padding 15%
```

---

## 9. Dataset yaml

예시:

```yaml
path: G:/aihub_dog_pose_yolo_2model_session/side
train: images/train
val: images/val

names:
  0: dog

kpt_shape: [22, 3]
flip_idx: [1, 0, 2, 8, 9, 10, 11, 12, 3, 4, 5, 6, 7, 13, 18, 19, 20, 21, 14, 15, 16, 17]
```

현재는 front/back과 side 각각 yaml을 따로 가진다.

```text
dog_pose_front_back_22kpt.yaml
dog_pose_side_22kpt.yaml
```

---

## 10. Session-level split

원본 AIHub split에는 유사 프레임 누수가 있을 수 있으므로, session-level split을 새로 만든다.

세션 기준:

```text
model_type + 기수 + 촬영날짜시간 + view
```

예:

```text
side / 1기 / 2024_08_30_17_12_30 / Left
```

같은 세션 안의 연속 프레임은 반드시 train 또는 val 중 한쪽에만 들어가야 한다.

현재 생성된 session split 예시:

```text
side model:
  train: 49,586 images
  val:   5,345 images
  session overlap: 0

front_back model:
  train: 51,933 images
  val:   6,619 images
  session overlap: 0
```

주의:

```text
현재 세션은 view까지 포함한다.
즉 같은 날짜/강아지라도 Left와 Right는 다른 세션으로 취급된다.
```

---

## 11. 주요 스크립트

로컬 재현에 필요한 주요 파일:

```text
2modle/keypoint_schemas.py
2modle/convert_2model_dataset.py
2modle/convert_session_split_dataset.py
2modle/check_session_leakage.py
2modle/validate_2model_dataset.py
2modle/train_2model.py
video_predict.py
validation_samples.py
```

각 파일 역할:

```text
keypoint_schemas.py
  22-keypoint schema, flip index, skeleton, label mapping 정의

convert_2model_dataset.py
  원본 AIHub JSON을 YOLO Pose 형식으로 변환

convert_session_split_dataset.py
  원본 train/validation을 무시하고 session-level split으로 변환

check_session_leakage.py
  변환된 dataset의 train/val 세션 겹침 여부 확인

validate_2model_dataset.py
  YOLO Pose 라벨 형식 검증

train_2model.py
  front_back 또는 side 모델 학습

video_predict.py
  영상 추론 및 skeleton 시각화

validation_samples.py
  validation 샘플 시각화 및 정성 검토
```

---

## 12. 학습 명령 예시

### Side model

```powershell
python 2modle/train_2model.py `
  --target side `
  --data 'G:\aihub_dog_pose_yolo_2model_session\side\dog_pose_side_22kpt.yaml' `
  --model yolo26s-pose.pt `
  --epochs 500 `
  --patience 30 `
  --imgsz 640 `
  --batch 8 `
  --device 0 `
  --workers 2 `
  --name 2model_side_s_session
```

### Front/back model

```powershell
python 2modle/train_2model.py `
  --target front_back `
  --data 'G:\aihub_dog_pose_yolo_2model_session\front_back\dog_pose_front_back_22kpt.yaml' `
  --model yolo26s-pose.pt `
  --epochs 500 `
  --patience 30 `
  --imgsz 640 `
  --batch 8 `
  --device 0 `
  --workers 2 `
  --name 2model_front_back_s_session
```

Windows에서 DataLoader worker 오류가 발생하면 `workers=0`으로 재개한다.

---

## 13. Resume

학습이 중간에 꺼졌을 경우:

```powershell
python -c "from ultralytics import YOLO; YOLO(r'runs\2model_side_s_session\weights\last.pt').train(resume=True, workers=0)"
```

front/back의 경우:

```powershell
python -c "from ultralytics import YOLO; YOLO(r'runs\2model_front_back_s_session\weights\last.pt').train(resume=True, workers=0)"
```

추론과 최종 평가는 일반적으로 `best.pt`를 사용한다.

---

## 14. 추론

추론 시 view에 따라 모델을 다르게 사용한다.

```text
Front / Back 영상 → front_back best.pt
Left / Right 영상 → side best.pt
```

`video_predict.py`는 다음을 수행한다.

```text
bbox 그리기
keypoint 점 그리기
skeleton line 그리기
left/right 색상 구분
confidence threshold 이하 keypoint 숨기기
```

중요:

```text
좌표를 후처리로 억지 보정하지 않는다.
모델이 낸 원본 좌표를 사용한다.
confidence 낮은 점만 숨긴다.
```

---

## 15. Opposite 라벨 개선 실험

현재 수정하고자 하는 것은 측면 모델이다.

전/후방 모델은 현재 체계를 유지한다.

측면 모델의 문제:

```text
기존 측면 모델은 촬영 방향 기준 라벨만 사용한다.
실제 보행 영상에서는 반대편 다리가 명확히 보이는 프레임이 있다.
이때 pose가 촬영 방향 쪽 다리가 아니라 반대편 다리에 튀는 현상이 발생할 수 있다.
```

개선 방향:

```text
명확하게 반대편 다리가 보이는 이미지에 대해 Opposite pose를 찍는다.
코드 단계에서 Opposite pose는 촬영 방향의 반대 방향으로 강제한다.
```

매핑:

```text
Left view:
  기본 라벨      → left keypoint
  Opposite 라벨 → right keypoint

Right view:
  기본 라벨      → right keypoint
  Opposite 라벨 → left keypoint
```

라벨링 주의사항:

```text
반드시 반대편 다리가 명확하게 보이는 경우에만 Opposite 라벨링한다.
헷갈리면 찍지 않는다.
가려진 경우 찍지 않는다.
가까운 쪽 다리와 겹쳐 구분이 어려우면 찍지 않는다.
관절 위치를 정확히 특정할 수 없으면 찍지 않는다.
```

---

## 16. Opposite 파일럿 실험 설계

파일럿 목적:

```text
Opposite 라벨 정책이 실제 동영상에서 pose가 반대편 다리로 튀는 현상을 줄이는지 확인한다.
```

학습 구성:

```text
fine-tune train:
  기존 side train 샘플 300장
  Opposite 수정 train 샘플 100장

fine-tune val:
  기존 side val 샘플 60장
  Opposite 수정 val 샘플 20장
```

Opposite 샘플 구성 권장:

```text
train:
  50개 측면 세션 × 2장 = 100장

val:
  10개 측면 세션 × 2장 = 20장
```

같은 세션에서 2장을 고를 때는 연속 프레임을 피한다.

```text
나쁜 예:
  00155, 00156

좋은 예:
  00155, 00185
```

데이터 누수 방지:

```text
train과 val 사이에 같은 세션이 들어가면 안 된다.
같은 이미지가 기존 라벨과 수정 라벨로 중복 포함되면 안 된다.
val 이미지는 fine-tuning train에 절대 포함하지 않는다.
수정 라벨이 있는 이미지는 기존 라벨을 대체한다.
```

---

## 17. 중요한 시행착오 및 판단

### 1. 12 keypoint가 아니라 22 keypoint가 필요했다

원본 라벨 종류는 12개지만, 실제 데이터에서는 좌우 paired landmark가 존재한다.  
이를 보존하기 위해 22-keypoint 구조를 설계했다.

### 2. 원본 train/validation split에는 누수 가능성이 있었다

같은 촬영 세션의 유사 프레임이 train/validation에 섞여 있었다.  
이를 해결하기 위해 session-level split을 새로 만들었다.

### 3. 단일 모델보다 view별 모델 분리가 더 적합했다

front/back과 side는 좌우 판단 방식이 다르므로 모델을 분리했다.

### 4. 측면 opposite leg 문제는 후처리보다 라벨 정책 문제에 가깝다

모델 출력 좌표를 소프트웨어로 억지 보정하기보다, Opposite 라벨을 명시하는 방식이 더 안전하다.

### 5. 400장 파일럿은 최종 모델이 아니라 가능성 검증용이다

기존 10만 장 데이터에 100장 Opposite를 단순 추가하면 효과가 약할 수 있다.  
따라서 기존 best 모델에서 작은 균형 데이터셋으로 fine-tuning하여 정책 효과를 먼저 확인한다.

---

## 18. 가장 조심해야 할 것

```text
keypoint 순서 바꾸면 안 됨
flip_idx 틀리면 안 됨
front/back 좌우 매핑 헷갈리면 안 됨
side에서 Opposite 라벨 정책 명확히 해야 함
같은 세션이 train/val에 섞이면 안 됨
같은 이미지가 기존 라벨과 수정 라벨로 중복 들어가면 안 됨
검증셋 이미지를 fine-tuning train에 넣으면 안 됨
```

---

## 한 줄 요약

이 프로젝트는 단순 YOLO Pose 학습 프로젝트가 아니라, AIHub 반려견 해부학 라벨을 22-keypoint schema로 재정의하고, view별 좌우 매핑과 session-level split을 설계하여 실제 보행 영상에서 안정적인 skeleton 추정을 목표로 하는 데이터 파이프라인 및 pose estimation 프로젝트다.
