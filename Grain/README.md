# Hybrid Rocket Grain Burning Area Simulator (Image + Arbitrary Port Layout)

이미지에서 **포트 중심 배치(원 중심)** 를 자동 검출하고, 사용자가 입력한 실제 치수(그레인 직경/길이, 포트 반지름, 산화제 질유량 등)를 기반으로
하이브리드 로켓(파라핀 연료 가정)의 **시간에 따른 연소면적(Burning Surface Area)** 과 **포트 단면적(Flow Area)** 변화를 시뮬레이션하는 코드입니다.

- 포트가 동심원 위 균일 배치가 아니어도 상관없음 (이미지에서 중심 좌표를 얻어 계산)
- 포트가 서로 합쳐지는 경우(합집합) 자동 반영 (마스크 기반)
- 외벽에 닿으면 연소면적이 0으로 떨어지도록 처리

> ⚠️ 주의: 현재 모델은 “기하(2D 단면) 기반” 단순화 모델이며,
> 실제 하이브리드 연소는 유동/열전달/혼합/산화제 분포 등에 의해 달라질 수 있습니다.

---

## Features

- ✅ 이미지 기반 포트 중심 좌표 추출 (OpenCV HoughCircles)
- ✅ 임의 배치 다중 포트 처리 (포트 합집합)
- ✅ marching squares 기반 서브픽셀 둘레 계산 (skimage)
- ✅ 시간에 따른 연소면적(둘레 × 그레인 길이) 계산
- ✅ 외벽 도달 시 그래프에서 연소면적 0 처리

---

## Requirements

- Python 3.9+ 권장
- 패키지:
  - numpy
  - matplotlib
  - opencv-python
  - scikit-image

설치:

```bash
pip install numpy matplotlib opencv-python scikit-image
