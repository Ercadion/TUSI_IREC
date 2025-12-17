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

def circle(x,y,cx=0,cy=0,r=1):
    return np.sqrt((x-cx)**2+(y-cy)**2)-r

def box(x,y,cx=0,cy=0,hx=1,hy=1):
    return np.maximum(np.abs(x-cx)-hx, np.abs(y-cy)-hy)

def U(a,b): return np.minimum(a,b)
def I(a,b): return np.maximum(a,b)
def D(a,b): return np.maximum(a,-b)

EXTRA = {
    "circle":circle, "box":box,
    "U":U, "I":I, "D":D
}

def eval_expr(expr,X,Y):
    expr = expr.replace("^","**")
    return eval(expr, {"__builtins__":{}},
                dict(SAFE, **EXTRA, x=X, y=Y))

# =========================
# marching squares 둘레
# =========================
def perimeter_ms(mask, dx):
    contours = measure.find_contours(mask.astype(float),0.5)
    p=0
    for c in contours:
        d=np.diff(c,axis=0)
        p+=np.sum(np.sqrt((d*d).sum(axis=1)))
    return p*dx

# =========================
# 시뮬레이터
# =========================
def simulate(exprs, xlim, ylim,
             D_grain, L_grain,
             m_dot_ox,
             t_end, dt=0.05,
             a=1.1706e-4, n=0.62,
             grid=1000):

    xs=np.linspace(xlim[0],xlim[1],grid)
    ys=np.linspace(ylim[0],ylim[1],grid)
    X,Y=np.meshgrid(xs,ys)
    dx=xs[1]-xs[0]

    scale=D_grain/(xlim[1]-xlim[0])  # m/unit
    dx*=scale

    Grain_outer = circle(X,Y,0,0,xlim[0])

    F=np.full_like(X,1e9)
    for e in exprs:
        F=np.minimum(F, eval_expr(e,X,Y))

    t_list=[]
    A_port_list=[]
    A_burn_list=[]
    snapshots=[]

    t=0
    while t<t_end:
        gas=F<=0
        A_port=gas.sum()*dx*dx
        if A_port<=0: break

        P=perimeter_ms(gas,dx)
        A_burn=P*L_grain

        G=m_dot_ox/A_port
        rdot=a*(G**n)

        F=F-rdot*dt

        t_list.append(t)
        A_port_list.append(A_port)
        A_burn_list.append(A_burn)

        if t==0 or t == t_end - dt:
            snapshots.append(gas.copy())

        t+=dt

    return t_list,A_port_list,A_burn_list,snapshots,dx,X,Y, Grain_outer

# =========================
# 메인
# =========================
if __name__=="__main__":
    n=int(input("방정식 개수: "))
    exprs=[input(f"{i+1}번 식: ") for i in range(n)]
    x0,x1=map(float,input("x 범위: ").split())
    y0,y1=map(float,input("y 범위: ").split())
    D=float(input("그레인 직경 [m]: "))
    L=float(input("그레인 길이 [m]: "))
    mox=float(input("산화제 질유량 [kg/s]: "))
    t_end=float(input("최대연소시간 [s]: "))

    t,Aport,Aburn,shots,dx,X,Y,Grain_outer=simulate(
        exprs,(x0,x1),(y0,y1),
        D,L,mox,t_end
    )

    # 그래프
    plt.figure()
    plt.plot(t,Aburn)
    plt.xlabel("Time [s]")
    plt.ylabel("Burning Area [m²]")
    plt.grid()

    plt.figure()
    plt.plot(t,Aport)
    plt.xlabel("Time [s]")
    plt.ylabel("Port Area [m²]")
    plt.grid()
    plt.show()

    # 단면 변화
    for i,g in enumerate(shots):
        plt.figure(figsize=(5,5))
        plt.contourf(X,Y,g,levels=[0,1])
        plt.contour(X, Y, Grain_outer, levels=[0], colors='k', linewidths=2)
        plt.gca().set_aspect("equal")
        plt.title(f"Grain Cross-section at t = {t[i]:.2f}s")
        plt.xlabel("x [coord]")
        plt.ylabel("y [coord]")
        plt.grid(True, alpha=0.3)
        plt.show()
