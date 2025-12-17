import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import measure


# -----------------------
# 외벽 원 찾기 (중심+반지름)
# -----------------------
def find_outer_circle(gray):
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    (cx, cy), R = cv2.minEnclosingCircle(cnt)
    return float(cx), float(cy), float(R)


# -----------------------
# 포트 중심만 추출 + "외벽 중심 기준으로 shift"
# -----------------------
def detect_port_centers_from_image(image_path, num_ports):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = img.copy()

    outer = find_outer_circle(gray)
    if outer is None:
        raise RuntimeError("외벽 원을 찾지 못했습니다.")
    cx_o, cy_o, R_outer_pix = outer

    # 포트 후보 검출(반지름은 정확히 안 써도 됨: 중심만 쓰려고)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(0.2 * R_outer_pix),
        param1=120,
        param2=35,
        minRadius=int(0.05 * R_outer_pix),
        maxRadius=int(0.65 * R_outer_pix),
    )
    if circles is None:
        raise RuntimeError("포트 원을 찾지 못했습니다.")

    c = np.round(circles[0]).astype(float)  # (x,y,r)

    # 중심에서 가까운 순으로 num_ports개 선택 (단일포트면 중앙에 가까운 원이 보통 정답)
    dx = c[:, 0] - cx_o
    dy = c[:, 1] - cy_o
    d = np.hypot(dx, dy)
    order = np.argsort(d)
    c = c[order[:num_ports]]

    # !!! 핵심: 이미지 좌표 -> 외벽 중심 기준 좌표로 shift (픽셀 단위)
    centers_pix_rel = np.stack([c[:, 0] - cx_o, c[:, 1] - cy_o], axis=1)

    return centers_pix_rel, R_outer_pix


# -----------------------
# 격자 생성 (물리좌표계: 그레인 중심이 (0,0))
# -----------------------
def make_grid(R_grain, grid_size):
    xs = np.linspace(-R_grain, R_grain, grid_size)
    ys = np.linspace(-R_grain, R_grain, grid_size)
    dx = xs[1] - xs[0]
    X, Y = np.meshgrid(xs, ys)
    mask_grain = (X**2 + Y**2) <= (R_grain**2)
    return X, Y, mask_grain, dx


def gas_mask_from_ports(centers, radii, X, Y, mask_grain):
    gas = np.zeros_like(X, dtype=bool)
    for (cx, cy), r in zip(centers, radii):
        gas |= ((X - cx)**2 + (Y - cy)**2) <= r*r
    gas &= mask_grain
    return gas


# -----------------------
# marching squares 둘레 (닫힌 contour 보정)
# -----------------------
def perimeter_from_mask_marching(gas, dx):
    contours = measure.find_contours(gas.astype(float), 0.5)
    per = 0.0
    for c in contours:
        if len(c) < 2:
            continue
        diffs = np.diff(c, axis=0)
        seg = np.sqrt((diffs**2).sum(axis=1)).sum()

        # 닫기(마지막->첫번째)
        close = np.linalg.norm(c[0] - c[-1])
        seg += close

        per += seg
    return per * dx


def simulate_hybrid_paraffin_general(
    centers, radii, grain_radius, grain_length,
    m_dot_ox, burn_time, dt=0.02,
    a=0.488, n=0.62,
    grid_size=1000,
):
    centers = np.asarray(centers, float)
    radii = np.asarray(radii, float)
    R = float(grain_radius)

    X, Y, mask_grain, dx = make_grid(R, grid_size)

    t_list, A_burn_list, A_port_list = [], [], []
    t = 0.0

    while t <= burn_time:
        gas = gas_mask_from_ports(centers, radii, X, Y, mask_grain)

        perimeter = perimeter_from_mask_marching(gas, dx)
        A_port = gas.sum() * dx * dx
        A_burn = perimeter * grain_length

        t_list.append(t)
        A_burn_list.append(A_burn)
        A_port_list.append(A_port)

        if A_burn <= 0 or A_port <= 0:
            break

        G = m_dot_ox / A_port
        r_dot = a * (G**n)
        radii = radii + r_dot * dt

        # 외벽 도달하면 다음 점에서 0으로 떨어뜨리기
        if np.any(np.hypot(centers[:, 0], centers[:, 1]) + radii >= R):
            t_list.append(t + dt)
            A_burn_list.append(0.0)
            A_port_list.append(A_port)
            break

        t += dt

    return np.array(t_list), np.array(A_burn_list), np.array(A_port_list)


def main():
    image_path = input("이미지 파일 경로: ").strip()
    num_ports = int(input("포트 개수: "))
    centers_pix_rel, R_outer_pix = detect_port_centers_from_image(image_path, num_ports)

    grain_diameter = float(input("그레인 직경 D_grain [m]: "))
    grain_radius = grain_diameter / 2.0

    # !!! 핵심: 그레인 반지름으로 스케일 잡기 (port 검출반지름에 의존 X)
    scale = grain_radius / R_outer_pix
    centers = centers_pix_rel * scale

    r_port = float(input("포트 실제 반지름 r_port [m]: "))
    radii = np.full(len(centers), r_port)

    grain_length = float(input("그레인 길이 L_grain [m]: "))
    m_dot_ox = float(input("산화제 질유량 m_dot_ox [kg/s]: "))
    burn_time = float(input("최대 연소 시간 [s]: "))

    t, A_burn, A_port = simulate_hybrid_paraffin_general(
        centers, radii, grain_radius, grain_length,
        m_dot_ox, burn_time,
        dt=0.02, a=3.0e-5, n=0.8, grid_size=1500
    )

    plt.figure()
    plt.plot(t, A_burn)
    plt.xlabel("Time [s]")
    plt.ylabel("Burning Surface Area [m$^2$]")
    plt.grid(True)

    plt.figure()
    plt.plot(t, A_port)
    plt.xlabel("Time [s]")
    plt.ylabel("Port Cross-sectional Area [m$^2$]")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
