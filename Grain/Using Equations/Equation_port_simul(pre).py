import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

# =========================
# 안전 eval
# =========================
SAFE = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "sqrt": np.sqrt, "abs": np.abs,
    "exp": np.exp, "log": np.log,
    "pi": np.pi,
    "min": np.minimum,
    "max": np.maximum,
}

def circle(x, y, cx=0, cy=0, r=1):
    return np.sqrt((x-cx)**2 + (y-cy)**2) - r

def box(x, y, cx=0, cy=0, hx=1, hy=1):
    return np.maximum(np.abs(x-cx)-hx, np.abs(y-cy)-hy)

def U(a,b): return np.minimum(a,b)
def I(a,b): return np.maximum(a,b)
def D(a,b): return np.maximum(a,-b)

EXTRA = {"circle":circle, "box":box, "U":U, "I":I, "D":D}

def eval_expr(expr, X, Y):
    expr = expr.replace("^", "**")
    return eval(expr, {"__builtins__":{}}, dict(SAFE, **EXTRA, x=X, y=Y))

# =========================
# F(연속장)에서 컨투어를 뽑아
# 서브픽셀 둘레/면적 계산
# =========================
def contours_to_phys_xy(c, x0, y0, dx_unit, dy_unit, scale):
    """
    c: (N,2) = (row, col) index 좌표
    -> 물리좌표 (m) 로 변환
    """
    rows = c[:, 0]
    cols = c[:, 1]
    x_unit = x0 + cols * dx_unit
    y_unit = y0 + rows * dy_unit
    x_m = x_unit * scale
    y_m = y_unit * scale
    return x_m, y_m

def perimeter_from_F(F_eff, x0, y0, dx_unit, dy_unit, scale):
    contours = measure.find_contours(F_eff, 0.0)
    P = 0.0
    for c in contours:
        x, y = contours_to_phys_xy(c, x0, y0, dx_unit, dy_unit, scale)
        pts = np.column_stack([x, y])
        if len(pts) < 3:
            continue
        pts2 = np.vstack([pts, pts[0]])  # 닫기
        d = np.diff(pts2, axis=0)
        P += np.sum(np.sqrt((d*d).sum(axis=1)))
    return P

def area_from_F(F_eff, x0, y0, dx_unit, dy_unit, scale):
    """
    0-레벨 컨투어(다각형)로 면적(포트 단면적)을 계산(서브픽셀)
    - 포트(음수 영역)가 여러 개면 면적 합산
    """
    contours = measure.find_contours(F_eff, 0.0)
    A = 0.0
    for c in contours:
        x, y = contours_to_phys_xy(c, x0, y0, dx_unit, dy_unit, scale)
        if len(x) < 3:
            continue
        # shoelace
        A_poly = 0.5 * np.abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))
        A += A_poly
    return A

# =========================
# 시뮬레이터
# =========================
def simulate(exprs, xlim, ylim,
             D_grain, L_grain,
             m_dot_ox,
             t_end=10.0, dt=0.05,
             a=1.1706e-4, n=0.62,
             grid=1000,
             n_snap=5):

    # 격자(좌표계 단위 = "coord")
    xs = np.linspace(xlim[0], xlim[1], grid)
    ys = np.linspace(ylim[0], ylim[1], grid)
    X, Y = np.meshgrid(xs, ys)

    dx_unit = xs[1] - xs[0]
    dy_unit = ys[1] - ys[0]

    # "x 범위(지름) == 실제 그레인 직경" 스케일
    # scale = (m / coord-unit)
    scale = D_grain / (xlim[1] - xlim[0])

    # 그레인 외벽: (0,0) 중심, coord 단위 반지름
    Rcoord = 0.5 * (xlim[1] - xlim[0])  # 지름/2
    grain_sdf = circle(X, Y, 0, 0, Rcoord)
    grain_mask = (grain_sdf <= 0)

    # 포트/가스 도형의 SDF F (여러 식이면 union=min)
    F = np.full_like(X, 1e9, dtype=float)
    for e in exprs:
        F = np.minimum(F, eval_expr(e, X, Y))

    # 그레인 밖에서는 의미 없으니 큰 양수로 마스킹 (컨투어가 밖에서 생기지 않게)
    def F_effective(F):
        return np.where(grain_mask, F, 1e9)

    t_list, A_port_list, A_burn_list, A_regression_list = [], [], [], []
    snapshots = []

    # 스냅샷 시간들(균일 샘플)
    snap_times = np.linspace(0, t_end, n_snap)

    t = 0.0
    snap_idx = 0
    tol = 0.5*dt

    while t <= t_end + 1e-12:
        F_eff = F_effective(F)

        # 포트 단면적(서브픽셀, m^2)
        A_port = area_from_F(F_eff, xlim[0], ylim[0], dx_unit, dy_unit, scale)
        if A_port <= 0:
            break

        # 연소면적(옆면) = 포트 경계 둘레 * 길이 (m^2)
        P = perimeter_from_F(F_eff, xlim[0], ylim[0], dx_unit, dy_unit, scale)
        A_burn = P * L_grain

        t_list.append(t)
        A_port_list.append(A_port)
        A_burn_list.append(A_burn)

        # 스냅샷 저장
        if snap_idx <= len(snap_times) and abs(t - snap_times[snap_idx]) <= tol:
            port_mask = (F_eff <= 0)
            solid_mask = grain_mask & (~port_mask)
            snapshots.append((t, solid_mask, port_mask))
            snap_idx += 1

        # regression
        G = m_dot_ox / A_port  # kg/(m^2 s)
        rdot = a * (G**n)      # m/s  (※ a,n 단위 일치 필요)
        A_regression_list.append(rdot)

        # SDF 가정 하에서 F를 rdot*dt만큼 "팽창" (F=0 경계를 밖으로 이동)
        F = F - (rdot * dt) / scale  # ★주의: F는 coord 단위, rdot*dt는 m -> coord로 환산

        # 외벽 도달(대략): 포트가 그레인 경계에 닿으면 종료(면적 0으로 떨굼)
        # 경계 픽셀 근처에서 True가 생기면 닿았다고 판단
        boundary = grain_mask & (
            np.roll(~grain_mask, 1, 0) | np.roll(~grain_mask, -1, 0) |
            np.roll(~grain_mask, 1, 1) | np.roll(~grain_mask, -1, 1)
        )
        port_mask_now = (F_eff <= 0)
        if np.any(port_mask_now & boundary):
            # 다음 스텝에 0으로 찍기
            t_list.append(min(t + dt, t_end))
            A_port_list.append(A_port)
            A_burn_list.append(0.0)
            break

        t += dt

    # 시각화용 물리좌표 그리드(m)
    X_m = X * scale
    Y_m = Y * scale
    grain_sdf_m = circle(X_m, Y_m, 0, 0, D_grain/2)

    return (np.array(t_list), np.array(A_port_list), np.array(A_burn_list), np.array(A_regression_list),
            snapshots, X_m, Y_m, grain_sdf_m)

# =========================
# 메인
# =========================
if __name__ == "__main__":
    n = int(input("방정식 개수: "))
    exprs = [input(f"{i+1}번 식: ") for i in range(n)]
    x0, x1 = map(float, input("x 범위: ").split())
    y0, y1 = map(float, input("y 범위: ").split())

    D = float(input("그레인 직경 [m]: "))
    L = float(input("그레인 길이(높이) [m]: "))
    mox = float(input("산화제 질유량 [kg/s]: "))
    t_end = float(input("최대연소시간 [s]: "))

    t, Aport, Aburn, Aregression, shots, X_m, Y_m, grain_sdf_m = simulate(
        exprs, (x0, x1), (y0, y1),
        D, L, mox,
        t_end=t_end, dt=0.05,
        a=1.1706e-4, n=0.62,
        grid=1000, n_snap=5
    )

    # 그래프
    plt.figure()
    plt.plot(t, Aburn)
    plt.xlabel("Time [s]")
    plt.ylabel("Burning Area [m²]")
    plt.grid(True)

    plt.figure()
    plt.plot(t, Aport)
    plt.xlabel("Time [s]")
    plt.ylabel("Port Area [m²]")
    plt.grid(True)

    plt.figure()
    plt.plot(t, Aregression)
    plt.xlabel("Time [s]")
    plt.ylabel("Regression Rate [m/s]")
    plt.grid(True)
    plt.show()

    # 단면 변화(연료/포트/외벽 같이 표시)
    for (ti, solid_mask, port_mask) in shots:
        plt.figure(figsize=(5, 5))
        # 연료(solid)만 채우기: levels를 2구간으로
        plt.contourf(X_m, Y_m, solid_mask.astype(float), levels=[-0.5, 0.5, 1.5], alpha=0.6)
        # 포트 경계
        plt.contour(X_m, Y_m, port_mask.astype(float), levels=[0.5], linewidths=2)
        # 그레인 외벽
        plt.contour(X_m, Y_m, grain_sdf_m, levels=[0], linewidths=2)

        plt.gca().set_aspect("equal")
        plt.title(f"Grain Cross-section at t = {ti:.2f}s")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.grid(True, alpha=0.3)
        plt.show()
