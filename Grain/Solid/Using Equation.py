def simulate_solid(
    exprs,                  # 포트 방정식(들)
    xlim, ylim,             # 좌표계 단위 범위
    D_grain, L_grain,       # 그레인 직경, 길이 [m]
    t_end=10.0, dt=0.05,    # 시간 설정
    grid=1000,              # 그리드 크기
    snapshot_dt=0.5,        # 스냅샷 간격 [s]

    # --- solid rocket options ---
    rdot_const=None,        # [m/s] 상수 연소율(권장: 가장 안전한 형태)
    rdot_func=None,         # callable: rdot = f(t, A_port, A_burn, P) [m/s]
    P_const=None,           # [Pa] 또는 [bar] 등 "사용자 정의 단위" 상수 압력(필요 시)
    a=None, n=None,         # 압력법칙 rdot=a*P^n 를 쓰고 싶을 때(값은 사용자 제공)
    # (선택) 질량 생성률 계산용
    rho_prop=None,          # [kg/m^3] 추진제 밀도(사용자 제공)
):
    # 격자(단위 = coord)
    xs = np.linspace(xlim[0], xlim[1], grid)
    ys = np.linspace(ylim[0], ylim[1], grid)
    X, Y = np.meshgrid(xs, ys)

    dx_unit = xs[1] - xs[0]
    dy_unit = ys[1] - ys[0]

    # scale = (m / coord-unit)
    scale = D_grain / (xlim[1] - xlim[0])

    # 그레인 외벽
    Rcoord = 0.5 * (xlim[1] - xlim[0])
    grain_sdf = circle(X, Y, 0.0, 0.0, Rcoord)
    grain_mask = (grain_sdf <= 0)

    # 포트/가스 도형의 SDF F (식이 여러 개이면 union=min)
    F = np.full_like(X, 1e9, dtype=float)
    for e in exprs:
        F = np.minimum(F, eval_expr(e, X, Y))

    # 그레인 밖은 무의미 -> 큰 양수 고정
    def F_effective(F_):
        return np.where(grain_mask, F_, 1e9)

    # 스냅샷 시간들
    if snapshot_dt is None or snapshot_dt <= 0:
        snap_times = np.array([0.0, t_end], dtype=float)
    else:
        snap_times = np.arange(0.0, t_end + 1e-12, snapshot_dt, dtype=float)
        if snap_times[-1] < t_end - 1e-12:
            snap_times = np.append(snap_times, t_end)
    snap_idx = 0
    tol = 0.5 * dt

    # 결과
    t_list, A_port_list, A_burn_list, rdot_list = [], [], [], []
    mgen_list = []  # 선택: 질량 생성률
    snapshots = []

    # rdot 계산 함수(우선순위: rdot_func > rdot_const > (a,n,P_const))
    def get_rdot(t, A_port, A_burn):
        if callable(rdot_func):
            return float(rdot_func(t, A_port, A_burn, P_const))
        if rdot_const is not None:
            return float(rdot_const)
        if (a is not None) and (n is not None) and (P_const is not None):
            # 단위 일관성은 사용자 책임(교육/형상 시뮬 용도)
            return float(a * (P_const ** n))
        raise ValueError("rdot 모델이 지정되지 않았습니다: rdot_const 또는 rdot_func 또는 (a,n,P_const)를 제공하세요.")

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

        # 고체 로켓: 연소율(후퇴율)
        rdot = get_rdot(t, A_port, A_burn)

        # 저장
        t_list.append(t)
        A_port_list.append(A_port)
        A_burn_list.append(A_burn)
        rdot_list.append(rdot)

        # (선택) 질량 생성률: m_dot_gen = rho * rdot * A_burn
        if rho_prop is not None:
            mgen_list.append(float(rho_prop) * rdot * A_burn)
        else:
            mgen_list.append(np.nan)

        # 스냅샷 저장
        while snap_idx < len(snap_times) and (t + tol) >= snap_times[snap_idx]:
            port_mask = (F_eff <= 0)
            snapshots.append((snap_times[snap_idx], port_mask))
            snap_idx += 1

        # SDF 기반으로 포트 팽창(연료 후퇴) : F는 coord 단위, rdot*dt는 m -> coord 환산
        F = F - (rdot * dt) / scale

        # 외벽 도달 체크(포트가 그레인 경계에 닿으면 종료)
        boundary = grain_mask & (
            np.roll(~grain_mask, 1, 0) | np.roll(~grain_mask, -1, 0) |
            np.roll(~grain_mask, 1, 1) | np.roll(~grain_mask, -1, 1)
        )
        if np.any((F_eff <= 0) & boundary):
            break

        t += dt

    # 시각화용 물리좌표
    X_m = X * scale
    Y_m = Y * scale
    grain_sdf_m = circle(X_m, Y_m, 0.0, 0.0, D_grain / 2.0)

    return (
        np.array(t_list),
        np.array(A_port_list),
        np.array(A_burn_list),
        np.array(rdot_list),
        np.array(mgen_list),
        snapshots,
        X_m, Y_m,
        grain_sdf_m
    )
