import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from skimage import measure
from scipy.ndimage import distance_transform_edt

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

def rebuild_SDF_from_mask(port_mask: np.ndarray, dx_mm: float, dy_mm: float) -> np.ndarray:
    dist_out = distance_transform_edt(~port_mask, sampling=(dy_mm, dx_mm))
    dist_in = distance_transform_edt(port_mask, sampling=(dy_mm, dx_mm))

    F_sdf = dist_out - dist_in
    return F_sdf

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

def grain_chamber_setting(D_c, L_c, D_g, D_gc, L_gs, N, Den_ideal, Den_ratio, T_ideal, eff_combustion):
    """
    그레인 계산
    ================================
    D_c: 챔버 직경(mm)
    L_c: 챔버 길이(mm)
    D_g: 그레인 직경(mm)
    D_gc: 코어 직경(mm)
    L_gs: 세그먼트 길이(mm)
    N: 세그먼트 개수
    Den_ratio: 연료 밀도 비율(0~1)
    """
    V_c = np.pi/4 * D_c**2 * L_c                    # 챔버 부피, mm^3
    L_g = L_gs * N                                  # 그레인 총 길이, mm
    V_g = np.pi/4 * D_g**2 * L_g                    # 그레인 총 부피, mm^3
    V_load = V_g/V_c                                # 챔버 대비 그레인 부피 비율
    Den_actual = Den_ideal * Den_ratio              # 그레인 밀도, g/cc
    m_grain = V_g * Den_actual * 1e-6               # 그레인 질량, kg
    A_be = N * 2 * np.pi()/4 * (D_g**2 - D_gc**2)   # 그레인 끝면 연소면적, mm^2
    A_bc = N * np.pi() * D_gc * L_gs                # 그레인 코어 연소면적, mm^2
    A_bg = A_be + A_bc                              # 그레인 총 연소면적, mm^2
    T_actual = T_ideal * eff_combustion             # 연소실 온도, K
    return V_c, L_g, V_g, V_load, Den_actual, m_grain, A_be, A_bc, A_bg, T_actual

def D_gc_calculation(R_dot, t_interval, D_gc):
    """
    그레인 코어 직경 계산
    ================================
    R_dot: 연소 속도(mm/s)
    t_interval: 시간 간격(s)
    D_gc: 코어 직경(mm)
    """
    D_gc_new = D_gc + 2 * R_dot * t_interval
    return D_gc_new

def L_g_calculation(R_dot, t_interval, L_g, N):
    """
    그레인 길이 계산
    ================================
    R_dot: 연소 속도(mm/s)
    t_interval: 시간 간격(s)
    L_g: 그레인 길이(mm)
    N: 세그먼트 개수
    """
    L_g_new = L_g - 2 * R_dot * t_interval * N
    return L_g_new

def V_g_calculation(D_g, D_gc, L_g):
    """
    그레인 부피 계산
    ================================
    D_g: 그레인 직경(mm)
    D_gc: 코어 직경(mm)
    L_g: 그레인 길이(mm)
    """
    V_g = np.pi/4 * (D_g**2 - D_gc**2) * L_g
    return V_g

def m_g_calculation(V_g, Den_actual):
    """
    그레인 질량 계산
    ================================
    V_g: 그레인 부피(mm^3)
    Den_actual: 그레인 밀도(g/cc)
    """
    m_grain = V_g * Den_actual * 1e-6
    return m_grain

def A_be_calculation(N, D_g, D_gc):
    """
    그레인 끝면 연소면적 계산
    ================================
    N: 세그먼트 개수
    D_g: 그레인 직경(mm)
    D_gc: 코어 직경(mm)
    """
    A_be = N * 2 * np.pi()/4 * (D_g**2 - D_gc**2)
    return A_be

def A_bc_calculation(D_gc, L_g):
    """
    그레인 코어 연소면적 계산
    ================================
    D_gc: 코어 직경(mm)
    L_g: 그레인 길이(mm)
    """
    A_bc = np.pi() * D_gc * L_g
    return A_bc

def A_bg_calculation(A_be, A_bc):
    """
    그레인 총 연소면적 계산
    ================================
    A_be: 그레인 끝면 연소면적(mm^2)
    A_bc: 그레인 코어 연소면적(mm^2)
    """
    A_bg = A_be + A_bc
    return A_bg

def Find_max_A_bg(r_dot_test, t_interval, D_gc, L_g, N, D_g):
    """
    최대 연소면적 찾기
    ================================
    r_dot_test: 테스트용 후퇴율(mm/s)
    t_interval: 시간 간격(s)
    D_gc: 코어 직경(mm)
    L_g: 그레인 길이(mm)
    N: 세그먼트 개수
    D_g: 그레인 직경(mm)
    """
    find_max = True
    A_bg_list = []
    while find_max:
        D_gc_new = D_gc_calculation(r_dot_test, t_interval, D_gc)
        L_g_new = L_g_calculation(r_dot_test, t_interval, L_g, N)
        A_be_new = A_be_calculation(N, D_g, D_gc_new)
        A_bc_new = A_bc_calculation(D_gc_new, L_g_new)
        A_bg_new = A_bg_calculation(A_be_new, A_bc_new)
        A_bg_list.append(A_bg_new)
        if len(A_bg_list) > 1 and A_bg_list[-1] < A_bg_list[-2]:
            find_max = False
    max_A_bg = max(A_bg_list)
    return max_A_bg

def m_dot_GEN_calculation(D_g, D_gc, L_g, R_dot, N, Den_actual, t_interval):
    """
    질유량 계산
    ================================
    D_g: 그레인 직경(mm)
    D_gc: 코어 직경(mm)
    L_g: 그레인 길이(mm)
    R_dot: 연소 속도(mm/s)
    Den_actual: 그레인 밀도(g/cc)
    t_interval: 시간 간격(s)
    """
    D_gc_new = D_gc_calculation(R_dot, t_interval, D_gc)
    L_g_new = L_g_calculation(R_dot, t_interval, L_g, N)
    V_g_new = V_g_calculation(D_g, D_gc_new, L_g_new)
    V_diff = V_g - V_g_new
    m_GEN = m_g_calculation(V_diff, Den_actual)
    m_dot_GEN = m_GEN / t_interval
    return m_dot_GEN, D_gc_new, L_g_new, V_g_new

def m_dot_NOZ_calculation(P_c, P_atm, A_NOZ_t, C_star):
    """
    노즐 질유량 계산
    ================================
    P_c: 연소실 압력(MPa)
    P_atm: 대기압(MPa)
    A_NOZ_t: 노즐 단면적(mm^2)
    C_star: 특성 속도(m/s)
    """
    m_dot_NOZ = (P_c - P_atm) * A_NOZ_t / C_star
    return m_dot_NOZ

def m_dot_REMAIN_calculation(m_dot_GEN, m_dot_NOZ):
    """
    남은 질유량 계산
    ================================
    m_dot_GEN: 생성 질유량(kg/s)
    m_dot_NOZ: 노즐 질유량(kg/s)
    """
    m_dot_REMAIN = m_dot_GEN - m_dot_NOZ
    return m_dot_REMAIN

def m_REMAIN_calculation(m_dot_REMAIN, t_interval):
    """
    챔버 잔류 추진제 질량 계산
    ================================
    m_dot_REMAIN: 남은 질유량(kg/s)
    t_interval: 시간 간격(s)
    """
    m_REMAIN = m_dot_REMAIN * t_interval
    return m_REMAIN

def Den_REMAIN_calculation(V_c, V_g, m_REMAIN):
    """
    챔버 잔류 추진제 밀도 계산
    ================================
    V_c: 챔버 부피(mm^3)
    V_g: 그레인 부피(mm^3)
    m_dot_REMAIN: 남은 질유량(kg/s)
    t_interval: 시간 간격(s)
    """
    V_free = V_c - V_g
    den_REMAIN = m_REMAIN / V_free
    return den_REMAIN

def R_dot_calculation(a, n, P_chamber):
    """
    후퇴율 계산
    ================================
    a: 후퇴율 상수 a(mm/s)
    n: 후퇴율 상수 n(MPa)
    P_chamber: 연소실 압력(MPa)
    """
    R_dot = a * (P_chamber ** n)
    return R_dot

def P_chamber_calculation(Den_REMAIN, R_specific, T_actual, P_atm):
    """
    연소실 압력 계산
    ================================
    Den_REMAIN: 챔버 잔류 추진제 밀도(g/cc)
    R_specific: 기체상수(J/(kg*K))
    T_actual: 연소실 온도(K)
    P_atm: 대기압(MPa)
    """
    P_chamber = Den_REMAIN * R_specific * T_actual + P_atm
    return P_chamber

def Kn_calculation(A_bg, A_NOZ_t):
    """
    Knudsen 수 계산
    ================================
    A_bg: 그레인 총 연소면적(mm^2)
    A_NOZ_t: 노즐 단면적(mm^2)
    """
    Kn = A_bg / A_NOZ_t
    return Kn

def D_gc_transfer(A_port):
    """
    포트 단면적에서 코어 직경 계산
    ================================
    A_port: 포트 단면적(mm^2)
    """
    D_gc = (4.0 * A_port / np.pi) ** 0.5
    return D_gc


def simulate_NOZ_t(
    D_gc, D_g,
    L_g,
    t_interval,
    Kn_max,
    r_dot_test   
):
    """
    노즐 단면적 계산
    ================================
    D_gc: 코어 직경(mm)
    D_g: 그레인 직경(mm)
    L_g: 그레인 길이(mm)
    t_interval: 시간 간격(s)
    Kn_max: 최대 Knudsen 수(무차원)
    r_dot_test: 테스트용 후퇴율(mm/s)
    """
    max_A_bg = Find_max_A_bg(r_dot_test, t_interval, D_gc, L_g, N, D_g)
    A_NOZ_t = max_A_bg / Kn_max
    return A_NOZ_t
    
    
def simulate(
    exprs,                  # 포트 방정식(들)
    xlim, ylim,             # 좌표계 단위 범위
    V_c,                    # 챔버 부피 [mm^3]
    D_g, D_gc,              # 그레인 직경, 코어 직경 [mm]
    N,                      # 세그먼트 개수
    L_g,                    # 그레인 길이 [mm]
    V_g,                    # 그레인 부피 [mm^3]
    den_actual,             # 그레인 밀도 [g/cc]
    A_NOZ_t,                # 노즐 단면적 [mm^2]
    R_specific,             # 기체상수 [J/(kg*K)]
    gamma,                  # 비열비
    C_star,                 # 특성 속도 [m/s]
    T_actual,               # 연소실 온도 [K]
    P_target,               # 목표 압력 [MPa]
    P_atm,                  # 초기 압력(대기압) Pc [MPa]
    e_NOZ,                  # 노즐 삭마 정도
    a, n,                   # 후퇴율 상수 a, n [mm/s], [MPa]
    t_end, t_interval,      # 시간 간격[s]
    grid,                   # 그리드 크기
    snapshot_dt             # 스냅샷 간격[s]
):
    # 격자(단위 = coord)
    xs = np.linspace(xlim[0], xlim[1], grid)
    ys = np.linspace(ylim[0], ylim[1], grid)
    X, Y = np.meshgrid(xs, ys)

    dx_unit = xs[1] - xs[0]
    dy_unit = ys[1] - ys[0]

    # x 범위(지름)가 곧 실제 그레인 직경이므로 스케일 계산
    # scale = (mm / coord-unit)
    scale = D_g / (xlim[1] - xlim[0])

     # 그레인 외벽: (0,0) 중심, coord 단위 반지름
    Rcoord = 0.5 * (xlim[1] - xlim[0])
    grain_sdf = circle(X, Y, 0.0, 0.0, Rcoord)
    grain_mask = (grain_sdf <= 0)

    # # 포트/가스 도형의 SDF F (식이 여러 개이면 union=min)
    F = np.full_like(X, 1e9, dtype=float)
    for e in exprs:
        F = np.minimum(F, eval_expr(e, X, Y))

    # mm per pixel
    dx_mm = dx_unit * scale
    dy_mm = dy_unit * scale

    # 재생성 관련 변수
    rebuild_interval = 45
    rebuild_counter = 0
    dr_accum = 0.0
    dr_threshold = dx_mm * 3.0

    # 그레인 밖에서는 의미 없으니 큰 양수로 고정
    def F_effective(F_):
        return np.where(grain_mask, F_, 1e9)

    # 스냅샷 snapshot_dt 간격으로 저장
    if snapshot_dt is None or snapshot_dt <= 0:
        snap_times = np.array([0.0, t_end], dtype=float)
    else:
        snap_times = np.arange(0.0, t_end + 1e-12, snapshot_dt, dtype=float)
        if snap_times[-1] < t_end - 1e-12:
            snap_times = np.append(snap_times, t_end)
    snap_idx = 0
    tol = 0.5 * t_interval

    # 결과 저장용 리스트 생성
    t_list = []
    D_gc_list, L_g_list = [], []
    A_be_list, A_bc_list, A_bg_list = [], [], []
    m_g_list, m_dot_GEN_list, m_dot_NOZ_list, m_dot_REMAIN_list = [], [], [], []
    den_REMAIN_list, P_c_list, R_dot_list = [], [], []
    Kn_list = []
    snapshot_list = []

    t = 0.0
    P_c = P_atm
    R_dot = R_dot_calculation(a, n, P_c)
    m_g = m_g_calculation(V_g, den_actual)
    m_g_list.append(m_g)
    m_REMAIN = 0.0

    while t <= t_end + 1e-12:
        F_eff = F_effective(F)

        # 포트 단면적(서브픽셀, mm^2)
        A_port = area_from_F(F_eff, xlim[0], ylim[0], dx_unit, dy_unit, scale)
        if A_port <= 0:
            break
        if t < 1e-12:
            D_port = D_gc_transfer(A_port)
            print("인식된 코어 직경 D_gc =", D_gc, "mm")
            print("오차율:", abs(D_port - D_gc) / D_gc * 100, "%")

        m_dot_GEN, D_gc, L_g, V_g =  m_dot_GEN_calculation(D_g, D_gc, L_g, R_dot, N, den_actual, t_interval)
        
        D_gc_list.append(D_gc)
        L_g_list.append(L_g)
        
        A_be = A_be_calculation(N, D_g, D_gc)
        A_bc = A_bc_calculation(D_gc, L_g)
        A_bg = A_bg_calculation(A_be, A_bc)
        Kn = Kn_calculation(A_bg, A_NOZ_t)
        A_be_list.append(A_be)
        A_bc_list.append(A_bc)
        A_bg_list.append(A_bg)
        Kn_list.append(Kn)

        m_g -= m_dot_GEN * t_interval
        m_g_list.append(m_g)

        m_dot_NOZ = m_dot_NOZ_calculation(P_c, P_atm, A_NOZ_t, C_star)
        m_dot_REMAIN = m_dot_REMAIN_calculation(m_dot_GEN, m_dot_NOZ)

        m_dot_GEN_list.append(m_dot_GEN)
        m_dot_NOZ_list.append(m_dot_NOZ)
        m_dot_REMAIN_list.append(m_dot_REMAIN)

        if m_dot_REMAIN < 0:
            m_dot_REMAIN = 0.0

        m_REMAIN += m_REMAIN_calculation(m_dot_REMAIN, t_interval)
        den_REMAIN = Den_REMAIN_calculation(V_c, V_g, m_REMAIN)
        
        den_REMAIN_list.append(den_REMAIN)

        P_c = P_chamber_calculation(den_REMAIN, R_specific, T_actual, P_atm)
        
        P_c_list.append(P_c)

        R_dot = R_dot_calculation(a, n, P_c)

        R_dot_list.append(R_dot)
        
        t_list.append(t)


        # 스냅샷 저장
        while snap_idx < len(snap_times) and (t + tol) >= snap_times[snap_idx]:
            snapshot_list.append((snap_times[snap_idx], F_eff.copy()))
            snap_idx += 1

        #SDF 가정 하에서 F를 rdot*dt만큼 팽창
        # F는 coord 단위, rdot*dt는 m -> coord로 환산
        #F = F - (r_dot * dt) / scale

        post_SDF_mask = (F_eff <= 0)
        dr_mm = R_dot * t_interval
        dr_accum += dr_mm
        rebuild_counter += 1
        
        if(rebuild_counter >= rebuild_interval) or (dr_accum >= dr_threshold):
            F = rebuild_SDF_from_mask(post_SDF_mask, dx_mm, dy_mm)
            F = np.where(grain_mask, F, 1e9)

            rebuild_counter = 0
            dr_accum = 0.0

        F = F - dr_mm

        F = np.where(grain_mask, F, 1e9)

        # 외벽 도달: 포트가 그레인 경계에 닿으면 종료
        # 경계 픽셀 근처에서 True가 생기면 닿았다고 판단
        if D_gc >= D_g or L_g <= 0 or m_g <= 0:
            if D_gc >= D_g:
                print(f"외벽 도달: D_gc={D_gc:.2f} mm >= D_g={D_g:.2f} mm")
            elif L_g <= 0:
                print(f"높이 도달: L_g={L_g:.2f} mm <= 0 mm")
            elif m_g <= 0:
                print(f"그레인 소진: m_g={m_g:.2f} g <= 0 g")
            break

        t += t_interval

    # 시각화용 물리좌표 그리드(m)
    X_m = X * scale
    Y_m = Y * scale
    grain_sdf_m = circle(X_m, Y_m, 0.0, 0.0, D_g / 2.0)

    return (
        np.array(t_list),
        np.array(D_gc_list), np.array(L_g_list),
        np.array(A_be_list), np.array(A_bc_list), np.array(A_bg_list),
        np.array(m_g_list), np.array(m_dot_GEN_list), np.array(m_dot_NOZ_list), np.array(m_dot_REMAIN_list),
        np.array(den_REMAIN_list), np.array(P_c_list), np.array(R_dot_list),
        np.array(Kn_list),
        snapshot_list,
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

    Den_grain = 1.841
    gamma = 1.137
    R_specific = 208.4
    T_ideal = 1600
    P_atm = 0.101
    C_star = 885
    e_NOZ = 0.0         #공식에 반영 안되어 있음
    Kn_max = 385



    n_expr = int(input("방정식 개수: ").strip())
    exprs = [input(f"{i+1}번 식: ").strip() for i in range(n_expr)]

    x0, x1 = map(float, input("x 범위 (예: -15 15): ").split())
    y0, y1 = map(float, input("y 범위 (예: -15 15): ").split())

    D_c = float(input("챔버 직경(mm): ").strip())
    L_c = float(input("챔버 길이(mm): ").strip())
    Type = input("연료 종류(KNDX, KNSB, KNSU, KNER, KNMN, KNFR, KNPSB): ").strip()
    D_g = float(input("그레인 직경(mm): ").strip())
    D_gc = float(input("코어 직경(mm): ").strip())
    L_gs = float(input("세그먼트 길이(mm): ").strip())
    N = int(input("세그먼트 개수: ").strip())
    Den_ratio = float(input("연료 밀도 비율(0~1): ").strip())
    P_target = float(input("목표 압력(MPa): ").strip())
    e = float(input("노즐 삭마 정도: ").strip())
    eff_combustion = float(input("연소 효율(0~1): ").strip())
    t_interval = float(input("시간 간격(s): ").strip())

    V_c, L_g, V_g, V_load, Den_actual, m_grain, A_be, A_bc, A_bg, T_actual = grain_chamber_setting(
        D_c, L_c, D_g, D_gc, L_gs, N, Den_grain, Den_ratio, T_ideal, eff_combustion)
    
    print("초기 조건은 다음과 같습니다.")
    print("연료 종류: KNSB")
    print(f"연소 효율: {eff_combustion:.2f}")
    print(f"챔버 직경: {D_c:.2f} mm")
    print(f"챔버 길이: {L_c:.2f} mm")
    print(f"챔버 부피: {V_c:.2f} mm^3")

    print(f"그레인 직경: {D_g:.2f} mm")
    print(f"코어 직경: {D_gc:.2f} mm")
    print(f"세그먼트 길이: {L_gs:.2f} mm")
    print(f"세그먼트 개수: {N}")
    print(f"그레인 길이: {L_g:.2f} mm")
    print(f"그레인 밀도: {Den_actual:.4f} g/cc")
    print(f"그레인 질량: {m_grain:.4f} kg")
    print(f"그레인 부피: {V_g:.2f} mm^3")
    print(f"챔버 대비 그레인 부피 비율: {V_load:.4f}")

    print(f"그레인 끝면 연소면적: {A_be:.2f} mm^2")
    print(f"그레인 코어 연소면적: {A_bc:.2f} mm^2")
    print(f"그레인 총 연소면적: {A_bg:.2f} mm^2")  

    print(f"목표 압력: {P_target:.2f} MPa")
    print(f"초기 압력(대기압): {P_atm:.3f} MPa")
    print(f"챔버 온도: {T_actual:.2f} K")
    print(f"기체상수: {R_specific:.2f} J/(kg*K)")
    print(f"비열비: {gamma:.3f}")
    print(f"특성 속도: {C_star:.2f} m/s")
    print(f"노즐 삭마 정도: {e:.2f}")

    print(f"시간 간격: {t_interval:.3f} s")

    check = input("시작하시겠습니까? (Y/N): ").strip().lower()
    if check == 'y':
        A_NOZ_t = simulate_NOZ_t(D_gc, D_g, L_g, t_interval, Kn_max, r_dot_test=0.5)
        print(f"계산된 노즐 단면적: {A_NOZ_t:.2f} mm^2")
        t, Dgc, Lg, Abe, Abc, Abg, mg, mdotGEN, mdotNOZ, mdotREMAIN, denREMAIN, Pc, Rdot, Kn_list, shots, Xmm, Ymm, grainsdfm = simulate(
            exprs, 
            (x0, x1), (y0, y1), 
            V_c,
            D_g, D_gc,  
            N,
            L_g,
            V_g, 
            Den_actual,
            A_NOZ_t, 
            R_specific, 
            gamma, 
            T_actual,
            P_target, 
            P_atm,
            e_NOZ, 
            a=0.5, n=0.5,
            t_end = 100.0, t_interval=t_interval,
            grid=1500,
            snapshot_dt = 0.05
        )
        plt.figure()
        plt.plot(t, Dgc, label="D_gc (코어 직경)")
        plt.xlabel("Time [s]")
        plt.ylabel("D_gc [mm]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, Lg, label="L_g (그레인 길이)")
        plt.xlabel("Time [s]")
        plt.ylabel("L_g [mm]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, Abe, label="A_be (끝면 연소면적)")
        plt.xlabel("Time [s]")
        plt.ylabel("A_be [mm²]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, Abc, label="A_bc (코어 연소면적)")
        plt.xlabel("Time [s]")
        plt.ylabel("A_bc [mm²]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, Abg, label="A_bg (총 연소면적)")
        plt.xlabel("Time [s]")
        plt.ylabel("A_bg [mm²]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, mg, label="m_g (그레인 질량)")
        plt.xlabel("Time [s]")
        plt.ylabel("m_g [kg]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, mdotGEN, label="m_dot_GEN (생성 질유량)")
        plt.xlabel("Time [s]")
        plt.ylabel("m_dot_GEN [kg/s]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, mdotNOZ, label="m_dot_NOZ (노즐 배출 질유량)")
        plt.xlabel("Time [s]")
        plt.ylabel("m_dot_NOZ [kg/s]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, mdotREMAIN, label="m_dot_REMAIN (남은 질유량)")
        plt.xlabel("Time [s]")
        plt.ylabel("m_dot_REMAIN [kg/s]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, denREMAIN, label="den_REMAIN (잔류 추진제 밀도)")
        plt.xlabel("Time [s]")
        plt.ylabel("den_REMAIN [g/cc]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, Pc, label="P_c (연소실 압력)")
        plt.xlabel("Time [s]")
        plt.ylabel("P_c [MPa]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, Rdot, label="R_dot (후퇴율)")
        plt.xlabel("Time [s]")
        plt.ylabel("R_dot [mm/s]")
        plt.grid(True)

        plt.figure()
        plt.plot(t, Kn_list, label="Kn (Knudsen 수)")
        plt.xlabel("Time [s]")
        plt.ylabel("Kn (무차원)")
        plt.grid(True)


        # 단면 변화(연료/포트/외벽 같이 표시)
        plt.figure(figsize=(7, 7))
    
        plt.contour(Xmm, Ymm, grainsdfm, levels=[0], colors="k", linewidths=2)
    
        cmap = plt.get_cmap("tab10")
        handles = []

        if len(shots) > 0:
            t0, F0 = shots[0]
            port0 = (F0 <= 0)
            solid0 = (grainsdfm <= 0) & (~port0)
            plt.contourf(XXmm, Ymm, solid0.astype(float),
                         levels=[-0.5, 0.5, 1.5], alpha=0.25)

        for i, (ti, Fsnap) in enumerate(shots):
            color = cmap(i % 10)
            plt.contour(
                Xmm, Ymm, 
                Fsnap, 
                levels=[0.0],
                colors=[color], 
                linewidths=1
            )
            line = mlines.Line2D([], [], color=color, linewidth=2, label=f"t={ti:.2f}s")
            handles.append(line)

        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("Port Evolution Over Time")
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.grid(True, alpha=0.3)
        plt.legend(handles=handles, loc="best")
        plt.show()
        print("시뮬레이션이 완료되었습니다.")
    else:
        print("시뮬레이션이 취소되었습니다.")
