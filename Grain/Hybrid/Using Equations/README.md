# Hybrid Rocket Grain Port Regression Simulator (2D SDF, Sub-pixel Contours)

수식으로 정의한 포트(가스 영역) 단면을 **SDF(Signed Distance Function)** 로 표현하고,  
하이브리드 로켓의 경험적 후퇴율 식

\[
\dot r = a\,G^n,\quad G=\frac{\dot m_{ox}}{A_{port}}
\]

을 이용해 시간에 따른 **포트 확장(regression)**, **포트 단면적**, **연소면적(둘레×길이)**, **후퇴율**을 계산/시각화하는 파이썬 시뮬레이터입니다.

> ⚠️ 본 코드는 교육/개념설계/형상 비교 목적의 단순 모델입니다.  
> 실제 엔진 설계에는 압력·혼합·열전달·3D 유동·연료/산화제 특성 및 실험 보정이 필요합니다.

---

## Features

- 포트 형상을 **수식(SDF)** 으로 입력 (`circle`, `box`, `star` + 집합연산 `U/I/D`)
- `skimage.measure.find_contours` 기반 **서브픽셀 둘레/면적 계산**
- 시간에 따라 포트 경계를 **rdot·dt 만큼 팽창**(SDF 이동)하여 연소 진행 모사
- 결과 그래프:
  - Burning Area (Perimeter × Length)
  - Port Area
  - Regression Rate
- 스냅샷 시간별 포트 경계 **오버레이 시각화**(Legend 포함)

---

## Requirements

- Python 3.9+ 권장
- Dependencies
  ```bash
  pip install numpy matplotlib scikit-image
