import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1) 기본 유틸 / 안전 eval
# -------------------------
SAFE = {
    # 수학 함수
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "arctan2": np.arctan2,
    "sqrt": np.sqrt, "abs": np.abs,
    "exp": np.exp, "log": np.log,
    "pi": np.pi, "e": np.e,

    # 조합(암시적 연산)
    "min": np.minimum,  # union에 사용 가능
    "max": np.maximum,  # intersection에 사용 가능
}

def _eval_expr(expr: str, X, Y, extra=None):
    expr = expr.replace("^", "**")
    scope = dict(SAFE)
    scope.update({"x": X, "y": Y})
    if extra:
        scope.update(extra)
    return eval(expr, {"__builtins__": {}}, scope)

# -------------------------
# 2) 암시적 도형 함수들
#    (f<=0 내부, f=0 경계)
# -------------------------
def circle(x, y, cx=0.0, cy=0.0, r=1.0):
    # 원: sqrt((x-cx)^2+(y-cy)^2) - r
    return np.sqrt((x - cx)**2 + (y - cy)**2) - r

def box(x, y, cx=0.0, cy=0.0, hx=1.0, hy=1.0, angle=0.0):
    """
    축 정렬/회전 사각형의 간단한 implicit (경계 표현용)
    hx,hy: 반가로/반세로(half extents)
    angle: 라디안. 0이면 축정렬.
    경계 근사: max(|x'|-hx, |y'|-hy) = 0
    """
    # 회전 좌표로 변환
    c, s = np.cos(angle), np.sin(angle)
    xp =  c*(x - cx) + s*(y - cy)
    yp = -s*(x - cx) + c*(y - cy)
    return np.maximum(np.abs(xp) - hx, np.abs(yp) - hy)

def star(x, y, cx=0.0, cy=0.0, r0=0.6, r1=1.0, k=5, angle=0.0, sharp=30.0):
    """
    k각 별(스파이크 k개)을 '극좌표 반지름 함수'로 만든 뒤
    f = rho - r(theta) 로 암시적 경계 생성.
    - r0: 안쪽 반지름(골)
    - r1: 바깥 반지름(꼭지)
    - k : 스파이크 개수
    - sharp: 클수록 뾰족(곡선이 더 급해짐). 10~60 정도 추천.
    """
    X = x - cx
    Y = y - cy
    th = np.arctan2(Y, X) - angle
    rho = np.sqrt(X*X + Y*Y)

    # [0,1]로 바뀌는 파형을 만들어 r0~r1 사이로 스케일
    # cos(k*theta)를 sharp로 조절해서 뾰족하게
    w = 0.5 * (1.0 + np.cos(k * th))
    w = w**(sharp/10.0)  # sharp 조절
    r = r0 + (r1 - r0) * w

    return rho - r

# 합집합/교집합/차집합 helper (문자열에서 쓰기 좋게)
def U(a, b):  # union
    return np.minimum(a, b)

def I(a, b):  # intersection
    return np.maximum(a, b)

def D(a, b):  # difference a\b
    return np.maximum(a, -b)

EXTRA_FUNCS = {
    "circle": circle,
    "box": box,
    "star": star,
    "U": U, "I": I, "D": D,
}

# -------------------------
# 3) 플로터
# -------------------------
def plot_implicit(exprs, xlim=(-2,2), ylim=(-2,2), grid=1000, fill=False, title="Implicit"):
    xs = np.linspace(xlim[0], xlim[1], grid)
    ys = np.linspace(ylim[0], ylim[1], grid)
    X, Y = np.meshgrid(xs, ys)

    plt.figure(figsize=(7,7))

    for expr in exprs:
        F = _eval_expr(expr, X, Y, extra=EXTRA_FUNCS)

        if fill:
            # 내부(f<=0) 채우기
            plt.contourf(X, Y, F, levels=[-1e9, 0], alpha=0.35)
            plt.contour(X, Y, F, levels=[0], linewidths=2)
        else:
            # 경계만
            plt.contour(X, Y, F, levels=[0], linewidths=2)

    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(*xlim); plt.ylim(*ylim)
    plt.grid(True, alpha=0.25)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    print("암시적 식 f(x,y) 입력. (f<=0 내부, f=0 경계)")
    print("사용 가능: circle(...), box(...), star(...), U(a,b), I(a,b), D(a,b)")
    print("예: U(circle(x,y,r=1), box(x,y,hx=0.7,hy=0.7))")
    n = int(input("식 개수 n: ").strip())
    exprs = [input(f"{i+1}번째 식: ").strip() for i in range(n)]
    x0, x1 = map(float, input("x 범위 (예: -2 2): ").split())
    y0, y1 = map(float, input("y 범위 (예: -2 2): ").split())
    fill = input("채움 표시? (y/n): ").strip().lower().startswith("y")

    plot_implicit(exprs, xlim=(x0, x1), ylim=(y0, y1), grid=1000, fill=fill, title="Implicit plot")
