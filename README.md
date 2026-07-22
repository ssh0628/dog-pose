# Dog Pose

## Labeling Tool
입력:

```text
raw_images/
  dog_0001.jpg
  dog_0002.jpg
  ...
```

저장 결과:

```text
labeled_data/
  image/
    Front/
      dog_0001.jpg
    Back/
    Left/
    Right/
  label/
    Front/
      dog_0001.json
    Back/
    Left/
    Right/
```

조작:
- `Direction`에서 현재 이미지의 방향 선택
- `Label Set`에서 일반 라벨은 `Main`, 반대편 다리 라벨은 `Opposite` 선택
- `Main`은 JSON의 `annotation_info`에 저장
- `Opposite`는 같은 JSON의 `opposite_annotation_info`에 저장
- `Opposite` 저장 시 `opposite_direction`에 반대 방향이 기록됨 (`Left -> Right`, `Right -> Left`)
- `Opposite`는 `Left`, `Right` 측면 이미지에서만 사용
- `Label Set`을 바꿔도 반대 세트의 기존 포인트는 회색 참고점으로 표시됨
- 키포인트 종류 선택 후 이미지 클릭으로 점 추가
- 이미지 위 번호는 찍은 순서가 아니라 AI Hub 키포인트 번호 `1~12`
- `Names: ON/OFF` 버튼으로 이미지 위 키포인트명 표시를 켜고 끔
- 기존 점 클릭 후 드래그로 이동
- 기존 점 더블클릭 또는 `Delete`로 삭제
- `Save` 또는 `Cmd/Ctrl + S`로 이미지 복사와 JSON 저장

측면 opposite 라벨링 주의사항:
- 반대편 다리가 명확하게 보이는 경우에만 `Opposite` 라벨링 수행
- 헷갈리면 찍지 않음
- `Front`, `Back`은 기존 체계를 유지하고 `Opposite`를 사용하지 않음

## Label Checker

저장된 라벨이 제대로 만들어졌는지 검사하고, 이미지 위에 점을 오버레이해서 확인합니다. main 라벨은 초록색, opposite 라벨은 주황색으로 표시됩니다.

```bash
uv run python check_labels.py --data-root labeled_data
```

리포트만 보고 싶으면 GUI를 열지 않고 실행할 수 있습니다.

```bash
uv run python check_labels.py --data-root labeled_data --report-only
```

검사 항목:
- `image/{direction}`와 `label/{direction}` 파일 짝 누락
- JSON 파싱 오류
- `annotation_info` 누락 또는 타입 오류
- `opposite_annotation_info` 타입 오류
- opposite 라벨이 있을 때 `opposite_direction`이 촬영 방향의 반대인지
- `x`, `y` 좌표가 숫자인지, `0~1` 범위인지
- 허용되지 않은 키포인트 label
- `image_info.filename`과 실제 파일명 불일치

### Json Structure
```json
{
    "image_info": {
        "filename": "잠실_2024_09_06_15_26_25_00043",
        "hospital": "잠실동물병원",
        "file_format": "jpg",
        "image_size": 1070,
        "device": "CAM",
        "resolution": "1920X1080"
    },

    "annotation_info": [
        {
            "x": "0.4909090909090909",
            "y": "0.7203065134099617",
            "label": "Distal lateral aspect of the fifth metatarsus"
        }, ... ], 

    "opposite_annotation_info": [
        {
            "x": "0.512",
            "y": "0.684",
            "label": "Femorotibial joint"
        }
    ],
    "opposite_direction": "Right",

    "pet_medical_record_info": [
        {
            "foot_position": "left",
            "value": 1
        },
        {
            "foot_position": "right",
            "value": 4
        }
    ],

    "sensor_values": [
        [
            255,
            255,
            1,
            0,
            ...
        ], ...
    ], 
    "timestamp": 1725603985,
    "size": "소형견",
    "severity": 4,
    "age": "노령견",
    "dog_type": "말티즈"
}
```

### annotation_info 
- x, y : 0 ~ 1 사이 값 (이미지 비율에 맞춰 변환 필요/이 데이터셋에서는 1920x1080 기준)
- label : 한 이미지 안에서 여러번 등장 가능

### pet_medical_record_info
- foot_position : "left" / "right" / "both" 
- value : 중증도
- pose만 찍는다면 상관 없음

### sensor_values 
- 뭔지 모르겠음
- pose만 찍는다면 상관 없음

## Side Opposite Pilot Fine-tuning

`dataset/side_opposite_pilot`은 기존 side 300/60장과 opposite 100/20장을 합친
YOLO Pose 파일럿 데이터셋입니다. 기존 side 모델의 `best.pt`를 초기 가중치로
사용하되, `resume`이 아닌 별도의 새 학습으로 실행합니다.

```bash
python training/finetune_side_opposite.py --device 0
```

기본 체크포인트 경로는 `models/side/best.pt`입니다. 다른 위치의 체크포인트를
사용할 때만 `--model /path/to/best.pt`를 지정합니다.

Opposite 라벨이 추가된 데이터 100/20장만으로 전이학습하려면 별도 데이터셋
`dataset/side_opposite_only`을 선택합니다.

```bash
python training/finetune_side_opposite.py \
  --dataset opposite-only \
  --device 0
```

기본 실행 순서:
- 기존 `best.pt`를 동일한 80장 validation set에서 먼저 평가
- 400장 train set으로 100 epoch fine-tuning
- 결과를 `runs/side_opposite_pilot_finetune*`에 별도 저장

데이터와 체크포인트 경로만 확인하고 학습하지 않으려면 `--check-only`를 사용합니다.

```bash
python training/finetune_side_opposite.py \
  --check-only
```

## Per-keypoint Pose Evaluation

파인튜닝 모델의 22개 키포인트별 OKS `mAP50-95`, `AP50`, `AP75`를 계산하고,
촬영 방향 기준 Original/Opposite 관절을 별도로 집계합니다.

```bash
python training/evaluate_keypoints.py \
  --model runs/side_opposite_only_finetune/weights/best.pt \
  --dataset opposite-only
```

결과는 `runs/keypoint_evaluation/<model>_<dataset>_val` 아래의 CSV와 JSON으로
저장됩니다. 키포인트별 값은 단일 키포인트 OKS로 계산한 사용자 정의 AP이며,
Ultralytics가 출력하는 전체 인스턴스 Pose mAP와는 구분해서 사용합니다.
