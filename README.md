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
- 키포인트 종류 선택 후 이미지 클릭으로 점 추가
- 기존 점 클릭 후 드래그로 이동
- 기존 점 더블클릭 또는 `Delete`로 삭제
- `Save` 또는 `Cmd/Ctrl + S`로 이미지 복사와 JSON 저장

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
