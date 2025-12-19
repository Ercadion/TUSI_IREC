import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from skimage import measure

# SAFE eval
SAFE = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "arctan2": np.arctan2,
    "sqrt": np.sqrt, "abs": np.abs,
    "exp": np.exp, "log": np.log,
    "pi": np.pi, "e": np.e,
    "min": np.minimum,
    "max": np.maximum,
}

# 원 방정식 함수
def circle(x, y, cx=0.0, cy=0.0, r=1.0):
    """
    원 방정식 함수
    ================================
    cx, cy      : 중심
    r           : 반지름
    """
    return np.sqrt((x - cx)**2 + (y - cy)**2) - r

# 직사각형 방정식 함수
def box(x, y, cx=0.0, cy=0.0, hx=1.0, hy=1.0):
    """
    직사각형 방정식 함수
    ================================
    cx, cy      : 중심
    hx          : x방향 길이
    hy          : y방향 길이
    """
    return np.maximum(np.abs(x - cx) - hx, np.abs(y - cy) - hy)

# 별 방정식 함수
def star(x, y, cx=0.0, cy=0.0, r_inner=0.5, r_outer=1.0, k=5, sharp=8.0, angle=0.0):
    """
    별 방정식 함수
    ================================
    cx, cy     : 중심
    r_inner    : 골 반지름
    r_outer    : 꼭지 반지름
    k          : 별 꼭지 개수
    sharp      : 뾰족함
    angle      : 회전 각도 (rad)
    """
    X = x - cx
    Y = y - cy
    rho = np.sqrt(X*X + Y*Y)
    theta = np.arctan2(Y, X) - angle

    # cos(k*theta) 기반 별 꼭지 만들기
    w = 0.5 * (1.0 + np.cos(k * theta))       # [0,1]
    w = w ** sharp                             # 뾰족함 조절
    r = r_inner + (r_outer - r_inner) * w
    return rho - r

# 집합 연산 함수
# 합집합
def U(a, b): return np.minimum(a, b)
# 교집합
def I(a, b): return np.maximum(a, b)
# 차집합
def D(a, b): return np.maximum(a, -b)

# 입력 -> 함수 매핑
EXTRA = {"circle": circle, "box": box, "star": star, "U": U, "I": I, "D": D}

def eval_expr(expr: str, X, Y):
    expr = expr.replace("^", "**")
    scope = dict(SAFE)
    scope.update(EXTRA)
    scope.update({"x": X, "y": Y})
    return eval(expr, {"__builtins__": {}}, scope)

# F에서 contour를 통해 서브픽셀 둘레/면적 계산
def contours_to_phys_xy(c, x0, y0, dx_unit, dy_unit, scale):
    """
    컨투어 좌표를 물리좌표로 변환
    ==========================================
    c           : (N,2) = (row, col) index 좌표
    x0          : x 좌표의 시작점 (좌표계 단위)
    y0          : y 좌표의 시작점 (좌표계 단위)
    dx_unit     : x 방향 단위 길이 (좌표계 단위)
    dy_unit     : y 방향 단위 길이 (좌표계 단위)
    scale       : 스케일 (m/coord-unit)
    """
    rows = c[:, 0]
    cols = c[:, 1]
    x_unit = x0 + cols * dx_unit
    y_unit = y0 + rows * dy_unit
    x_m = x_unit * scale
    y_m = y_unit * scale
    return x_m, y_m

def perimeter_from_F(F_eff, x0, y0, dx_unit, dy_unit, scale):
    """
    0-레벨 컨투어(다각형)로 둘레(포트 단면 둘레) 계산(서브픽셀)
    포트(음수 영역)가 여러 개면 둘레 합산
    ==========================================
    F_eff: 2D array
    x0: x 좌표의 시작점 (좌표계 단위)
    y0: y 좌표의 시작점 (좌표계 단위)
    dx_unit: x 방향 단위 길이 (좌표계 단위)
    dy_unit: y 방향 단위 길이 (좌표계 단위)
    scale: 스케일 (m/coord-unit)
    """
    contours = measure.find_contours(F_eff, 0.0)
    P = 0.0
    for c in contours:
        x, y = contours_to_phys_xy(c, x0, y0, dx_unit, dy_unit, scale)
        if len(x) < 2:
            continue
        pts = np.column_stack([x, y])
        
        # 페곡선으로 가정해 마지막-처음 연결
        pts2 = np.vstack([pts, pts[0]])
        d = np.diff(pts2, axis=0)
        P += np.sum(np.sqrt((d*d).sum(axis=1)))
    return P

def area_from_F(F_eff, x0, y0, dx_unit, dy_unit, scale):
    """
    0-레벨 컨투어(다각형)로 면적(포트 단면적)을 계산(서브픽셀)
    포트(음수 영역)가 여러 개면 면적 합산
    ==========================================
    F_eff: 2D array
    x0: x 좌표의 시작점 (좌표계 단위)
    y0: y 좌표의 시작점 (좌표계 단위)
    dx_unit: x 방향 단위 길이 (좌표계 단위)
    dy_unit: y 방향 단위 길이 (좌표계 단위)
    scale: 스케일 (m/coord-unit)
    """
    contours = measure.find_contours(F_eff, 0.0)
    A = 0.0
    for c in contours:
        x, y = contours_to_phys_xy(c, x0, y0, dx_unit, dy_unit, scale)
        if len(x) < 3:
            continue
        # 다각형 면적 계산 (신발끈 공식)
        A_poly = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        A += A_poly
    return A

# 시뮬레이터
def simulate(
    exprs,                  # 포트 방정식(들)
    xlim, ylim,             # 좌표계 단위 범위
    D_grain, L_grain,       # 그레인 직경, 길이 [m]
    m_dot_ox,               # 산화제 질유량 [kg/s]
    t_end=10.0, dt=0.05,    # 시간 설정
    a=1.1706e-4, n=0.62,    # 단위 일치시키기 위해 a 값 변환하여 사용(0.488 -> 1.1706e-4)
    grid=1000,              # 그리드 크기
    n_snap=5                # 변화 스냅샷 개수
):
    # 격자(단위 = coord)
    xs = np.linspace(xlim[0], xlim[1], grid)
    ys = np.linspace(ylim[0], ylim[1], grid)
    X, Y = np.meshgrid(xs, ys)

    dx_unit = xs[1] - xs[0]
    dy_unit = ys[1] - ys[0]

    # x 범위(지름)가 곧 실제 그레인 직경이므로 스케일 계산
    # scale = (m / coord-unit)
    scale = D_grain / (xlim[1] - xlim[0])

     # 그레인 외벽: (0,0) 중심, coord 단위 반지름
    Rcoord = 0.5 * (xlim[1] - xlim[0])
    grain_sdf = circle(X, Y, 0.0, 0.0, Rcoord)
    grain_mask = (grain_sdf <= 0)

    # # 포트/가스 도형의 SDF F (식이 여러 개이면 union=min)
    F = np.full_like(X, 1e9, dtype=float)
    for e in exprs:
        F = np.minimum(F, eval_expr(e, X, Y))

    # 그레인 밖에서는 의미 없으니 큰 양수로 고정
    def F_effective(F_):
        return np.where(grain_mask, F_, 1e9)

    # 스냅샷 시간들(균일 샘플)
    snap_times = np.linspace(0.0, t_end, n_snap)
    snap_idx = 0
    tol = 0.5 * dt

    # 결과 저장용 리스트 생성
    t_list, A_port_list, A_burn_list, rdot_list = [], [], [], []
    snapshots = []

    t = 0.0
    while t <= t_end + 1e-12:
        F_eff = F_effective(F)

        # 포트 단면적(서브픽셀, m^2)
        A_port = area_from_F(F_eff, xlim[0], ylim[0], dx_unit, dy_unit, scale)
        if A_port <= 0:
            break
        
        # 연소면적(옆면) = 포트 경계 둘레 * 길이 (m^2)
        P = perimeter_from_F(F_eff, xlim[0], ylim[0], dx_unit, dy_unit, scale)
        A_burn = P * L_grain

        # 후퇴율(m/s)
        G = m_dot_ox / A_port
        rdot = a * (G ** n)

        # 저장 (길이 항상 동일)
        t_list.append(t)
        A_port_list.append(A_port)
        A_burn_list.append(A_burn)
        rdot_list.append(rdot)

        # 스냅샷 저장
        if snap_idx < len(snap_times) and abs(t - snap_times[snap_idx]) <= tol:
            port_mask = (F_eff <= 0)
            snapshots.append((t, port_mask))
            snap_idx += 1

        #SDF 가정 하에서 F를 rdot*dt만큼 팽창
        # F는 coord 단위, rdot*dt는 m -> coord로 환산
        F = F - (rdot * dt) / scale

        # 외벽 도달: 포트가 그레인 경계에 닿으면 종료
        # 경계 픽셀 근처에서 True가 생기면 닿았다고 판단
        boundary = grain_mask & (
            np.roll(~grain_mask, 1, 0) | np.roll(~grain_mask, -1, 0) |
            np.roll(~grain_mask, 1, 1) | np.roll(~grain_mask, -1, 1)
        )
        if np.any((F_eff <= 0) & boundary):
            break

        t += dt

    # 시각화용 물리좌표 그리드(m)
    X_m = X * scale
    Y_m = Y * scale
    grain_sdf_m = circle(X_m, Y_m, 0.0, 0.0, D_grain / 2.0)

    return (
        np.array(t_list),
        np.array(A_port_list),
        np.array(A_burn_list),
        np.array(rdot_list),
        snapshots,
        X_m, Y_m,
        grain_sdf_m
    )

# 메인
if __name__ == "__main__":
    print("f(x,y) <= 0 이 포트(가스) 내부입니다.")
    print("사용 가능: circle, box, star, U, I, D")
    print("예) circle(x,y,0,0,3)")
    print("예) U(circle(x,y,5,5,3), circle(x,y,-5,5,3))")
    print("예) star(x,y,0,0, r_inner=1.5, r_outer=3.0, k=5, sharp=8)")

    n_expr = int(input("방정식 개수: ").strip())
    exprs = [input(f"{i+1}번 식: ").strip() for i in range(n_expr)]

    x0, x1 = map(float, input("x 범위 (예: -15 15): ").split())
    y0, y1 = map(float, input("y 범위 (예: -15 15): ").split())

    D = float(input("그레인 직경 D_grain [m]: "))
    L = float(input("그레인 길이(높이) L_grain [m]: "))
    mox = float(input("산화제 질유량 m_dot_ox [kg/s]: "))
    t_end = float(input("최대연소시간 t_end [s]: "))


    t, Aport, Aburn, rdot, shots, X_m, Y_m, grain_sdf_m = simulate(
        exprs,
        (x0, x1), (y0, y1),
        D, L, mox,
        t_end=t_end, dt=0.05,
        a=1.1706e-4, n=0.62,
        grid=1000,
        n_snap=5
    )

    # 그래프
    plt.figure()
    plt.plot(t, Aburn)
    plt.xlabel("Time [s]")
    plt.ylabel("Burning Area [m²] (Perimeter × Length)")
    plt.grid(True)

    plt.figure()
    plt.plot(t, Aport)
    plt.xlabel("Time [s]")
    plt.ylabel("Port Area [m²] (Union)")
    plt.grid(True)

    plt.figure()
    plt.plot(t, rdot)
    plt.xlabel("Time [s]")
    plt.ylabel("Regression Rate rdot [m/s]")
    plt.grid(True)

    # 단면 변화(연료/포트/외벽 같이 표시)
    plt.figure(figsize=(7, 7))
    if len(shots) > 0:
        t0, port0 = shots[0]
        solid0 = (grain_sdf_m <= 0) & (~port0)
        plt.contourf(X_m, Y_m, solid0.astype(float),
                     levels=[-0.5, 0.5, 1.5], alpha=0.25)


    plt.contour(X_m, Y_m, grain_sdf_m, levels=[0], colors="k", linewidths=2)

    cmap = plt.get_cmap("tab10")
    handles = []
    for i, (ti, port_mask) in enumerate(shots):
        color = cmap(i % 10)
        plt.contour(
            X_m, Y_m, 
            port_mask.astype(float), 
            levels=[0.5],
            colors=[color], 
            linewidths=2
        )
        line = mlines.Line2D([], [], color=color, linewidth=2, label=f"t={ti:.2f}s")
        handles.append(line)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Port Evolution Over Time (Overlay)")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(True, alpha=0.3)
    plt.legend(handles=handles, loc="best")
    plt.show()
