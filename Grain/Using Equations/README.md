# Hybrid Rocket Grain Regression Simulator (2D SDF-based)

이 저장소는 **하이브리드 로켓 연료 그레인(port) 단면 형상**을  
**Signed Distance Function (SDF)** 기반으로 정의하고,  
연소에 따른 **포트 확대(regression)**, **포트 단면적**, **연소면적**, **후퇴율**을
시간에 따라 계산·시각화하는 파이썬 시뮬레이터입니다.

> 🔬 연구/설계 목적의 **준정량적 모델**이며, 실제 엔진 설계에는  
> 실험 보정 및 3D 효과, 열전달, 압력 변화 모델이 추가로 필요합니다.

---

## ✨ 주요 기능

- 임의의 포트 형상을 **수식(SDF)** 으로 정의
- Union / Intersection / Difference 연산 지원
- 서브픽셀 정확도의
  - 포트 단면적 (m²)
  - 연소면적 (m²)
  - 후퇴율 (m/s)
- 연소 시간에 따른
  - 포트 확장 시뮬레이션
  - 단면 변화 스냅샷 자동 저장
- 하이브리드 로켓 경험식
  \[
  \dot r = a \, G^n, \quad G=\frac{\dot m_{ox}}{A_{port}}
  \]

---

## 📦 요구 환경

- Python ≥ 3.9
- 필수 라이브러리
  ```bash
  pip install numpy matplotlib scikit-image
