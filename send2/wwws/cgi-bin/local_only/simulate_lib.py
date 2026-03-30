from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import ast

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from openpyxl import Workbook
from scipy.integrate import solve_ivp

ALLOWED_FUNCTIONS = {
    'abs': abs,
    'min': min,
    'max': max,
    'pow': pow,
    'sqrt': np.sqrt,
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'hypot': np.hypot,
}
ALLOWED_NAMES = {'x', 'y', 'vx', 'vy', 'G', 'M', 'k', 'c'} | set(ALLOWED_FUNCTIONS)


def _check_expression(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        _check_expression(node.body)
    elif isinstance(node, ast.BinOp):
        _check_expression(node.left)
        _check_expression(node.right)
    elif isinstance(node, ast.UnaryOp):
        _check_expression(node.operand)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
            raise ValueError(f'許可されていない関数: {ast.dump(node.func)}')
        for arg in node.args:
            _check_expression(arg)
    elif isinstance(node, ast.Name):
        if node.id not in ALLOWED_NAMES:
            raise ValueError(f'許可されていない変数: {node.id}')
    elif isinstance(node, ast.Constant):
        return
    else:
        raise ValueError(f'許可されていない構文: {type(node).__name__}')


def make_force_function(expr: str) -> ForceComponent:
    expr = expr.strip()
    if expr == '':
        raise ValueError('空の関数定義は許可されていません。')

    try:
        parsed = ast.parse(expr, mode='eval')
        _check_expression(parsed)
        compiled = compile(parsed, '<force>', 'eval')
    except SyntaxError as exc:
        raise ValueError(f'構文エラー: {exc}') from exc

    def force(x: float, y: float, vx: float, vy: float, params: Params) -> float:
        local_vars = {
            'x': x,
            'y': y,
            'vx': vx,
            'vy': vy,
            'G': params.G,
            'M': params.M,
            'k': params.k,
            'c': params.c,
            **ALLOWED_FUNCTIONS,
        }
        return float(eval(compiled, {'__builtins__': {}}, local_vars))

    return force




def fx_gravity(x: float, y: float, vx: float, vy: float, params: Params) -> float:
    r = np.hypot(x, y)
    if r < 1e-12:
        return 0.0
    return -params.G * params.M * x / (r**3)


def fy_gravity(x: float, y: float, vx: float, vy: float, params: Params) -> float:
    r = np.hypot(x, y)
    if r < 1e-12:
        return 0.0
    return -params.G * params.M * y / (r**3)


def fx_linear_spring(x: float, y: float, vx: float, vy: float, params: Params) -> float:
    return -params.k * x - params.c * vx


def fy_linear_spring(x: float, y: float, vx: float, vy: float, params: Params) -> float:
    return -params.k * y - params.c * vy

def fx_3(x: float, y: float, vx: float, vy: float, params: Params) -> float:
    r = np.hypot(x, y)
    return - x **3


def fy_3(x: float, y: float, vx: float, vy: float, params: Params) -> float:
    return - y **3



@dataclass
class Params:
    m: float
    M: float
    G: float = 1.0
    k: float = 0.0
    c: float = 0.0
    fx: ForceComponent = field(default_factory=lambda: fx_gravity)
    fy: ForceComponent = field(default_factory=lambda: fy_gravity)


@dataclass
class Initials:
    x0: float
    y0: float
    vx0: float
    vy0: float

ForceComponent = Callable[[float, float, float, float, Params], float]


def acceleration(x: float, y: float, vx: float, vy: float, params: Params) -> tuple[float, float]:
    """指定された fx / fy を使って加速度を計算する。"""
    ax = params.fx(x, y, vx, vy, params)
    ay = params.fy(x, y, vx, vy, params)
    return ax, ay


def rhs(t: float, state: np.ndarray, params: Params) -> np.ndarray:
    """運動方程式の右辺: [dx/dt, dy/dt, dvx/dt, dvy/dt] を返す。"""
    x, y, vx, vy = state
    ax, ay = acceleration(x, y, vx, vy, params)
    return np.array([vx, vy, ax, ay])


def solve_motion(initials: Initials, params: Params, t_span: tuple[float, float], num_points: int = 500) -> solve_ivp:
    """solve_ivp を使って 2 次元運動方程式を解く。"""
    y0 = np.array([initials.x0, initials.y0, initials.vx0, initials.vy0])
    t_eval = np.linspace(t_span[0], t_span[1], num_points)

    solution = solve_ivp(
        fun=lambda t, y: rhs(t, y, params),
        t_span=t_span,
        y0=y0,
        method='RK45',
        t_eval=t_eval,
        dense_output=True,
        rtol=1e-8,
        atol=1e-10,
    )
    return solution


def print_solution(solution: solve_ivp) -> None:
    """解の概要を表示する。"""
    print(f'成功: {solution.success}')
    print(f'ステータス: {solution.status}')
    print(f'計算ステップ数: {solution.nfev}')
    print(f'最終時刻: {solution.t[-1]:.6f}')
    print('最初と最後の位置:')
    print(f'  t=0: x={solution.y[0,0]:.6f}, y={solution.y[1,0]:.6f}')
    print(f'  t={solution.t[-1]:.6f}: x={solution.y[0,-1]:.6f}, y={solution.y[1,-1]:.6f}')


def write_excel(filename: str, solution: solve_ivp) -> None:
    """Excel ファイルに軌道を出力する。"""
    wb = Workbook()
    ws = wb.active
    ws.title = 'Trajectory'
    ws.append(['t', 'x', 'y', 'vx', 'vy'])
    for t, x, y, vx, vy in zip(solution.t, solution.y[0], solution.y[1], solution.y[2], solution.y[3]):
        ws.append([float(t), float(x), float(y), float(vx), float(vy)])
    wb.save(filename)


def plot_results(solution: solve_ivp, prefix: str = 'trajectory') -> None:
    """時間発展と配位空間のグラフを作成して保存する。"""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].plot(solution.t, solution.y[0], label='x(t)')
    axs[0, 0].plot(solution.t, solution.y[1], label='y(t)')
    axs[0, 0].set_xlabel('t')
    axs[0, 0].set_ylabel('position')
    axs[0, 0].legend()
    axs[0, 0].set_title('Position vs Time')

    axs[0, 1].plot(solution.y[0], solution.y[1], color='blue')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y')
    axs[0, 1].set_title('Trajectory in x-y plane')
    axs[0, 1].grid(True)

    axs[1, 0].plot(solution.y[0], solution.y[2], label='x-vx')
    axs[1, 0].plot(solution.y[1], solution.y[3], label='y-vy')
    axs[1, 0].set_xlabel('position')
    axs[1, 0].set_ylabel('velocity')
    axs[1, 0].legend()
    axs[1, 0].set_title('Phase-space Trajectory')
    axs[1, 0].grid(True)

    axs[1, 1].plot(solution.t, solution.y[2], label='vx(t)')
    axs[1, 1].plot(solution.t, solution.y[3], label='vy(t)')
    axs[1, 1].set_xlabel('t')
    axs[1, 1].set_ylabel('velocity')
    axs[1, 1].legend()
    axs[1, 1].set_title('Velocity vs Time')
    axs[1, 1].grid(True)

    fig.tight_layout()
    fig.savefig(f'{prefix}_plots.png', dpi=200)
    plt.close(fig)


def animate_trajectory(solution: solve_ivp, filename: str = 'trajectory.gif', fps: int = 10, max_frames: int = 80) -> None:
    """x-y 平面のアニメーションを作成して GIF で保存する。"""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(solution.y[0], solution.y[1], color='gray', alpha=0.5)
    line, = ax.plot([], [], 'ro-', lw=2)
    point, = ax.plot([], [], 'ro')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('2D Trajectory Animation')
    ax.grid(True)

    ax.set_xlim(np.min(solution.y[0]) * 1.1, np.max(solution.y[0]) * 1.1)
    ax.set_ylim(np.min(solution.y[1]) * 1.1, np.max(solution.y[1]) * 1.1)

    frame_indices = np.linspace(0, len(solution.t) - 1, min(len(solution.t), max_frames), dtype=int)

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def update(index):
        frame = frame_indices[index]
        xdata = solution.y[0, :frame + 1]
        ydata = solution.y[1, :frame + 1]
        line.set_data(xdata, ydata)
        point.set_data([float(solution.y[0, frame])], [float(solution.y[1, frame])])
        return line, point

    anim = FuncAnimation(fig, update, frames=len(frame_indices), init_func=init, blit=True, interval=1000/fps)
    writer = PillowWriter(fps=fps)
    anim.save(filename, writer=writer)
    plt.close(fig)


def main() -> None:
    params = Params(
        m=1.0,
        M=1.0,
        G=1.0,
        k=2.0,
        c=0,
        # fx=fx_linear_spring,
        # fy=fy_linear_spring,
        fx=fx_3,
        fy=fy_3,
    )

    ### 力を定義　
    # params.fx = fx_gravity
    # params.fy = fy_gravity    
    # params.fx = fx_linear_spring
    # params.fy = fy_linear_spring
    ###

    initials = Initials(x0=1.0, y0=0.0, vx0=-2, vy0=1.0)
    t_span = (0.0, 10.0)

    solution = solve_motion(initials, params, t_span, num_points=1000)
    print_solution(solution)
    write_excel('trajectory.xlsx', solution)
    plot_results(solution, prefix='trajectory')
    animate_trajectory(solution, filename='trajectory.gif', fps=8, max_frames=60)
    print('\ntrajectory.xlsx に出力しました。')
    print('trajectory_plots.png と trajectory.gif を作成しました。')


if __name__ == '__main__':
    main()
