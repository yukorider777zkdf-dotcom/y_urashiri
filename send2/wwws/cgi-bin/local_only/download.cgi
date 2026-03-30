#!/usr/bin/env python3
import json
import os
import sys
from io import BytesIO

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_ONLY_PATHS = [
    HERE,
    os.path.abspath(os.path.join(HERE, 'local_only')),
    os.path.abspath(os.path.join(HERE, '..', 'local_only')),
]
for LOCAL_ONLY in LOCAL_ONLY_PATHS:
    if os.path.isdir(LOCAL_ONLY) and LOCAL_ONLY not in sys.path:
        sys.path.insert(0, LOCAL_ONLY)
        break

from openpyxl import Workbook
from simulate_lib import (
    Initials,
    Params,
    solve_motion,
    fx_gravity,
    fy_gravity,
    fx_linear_spring,
    fy_linear_spring,
    fx_3,
    fy_3,
    make_force_function,
)

MODEL_FUNCS = {
    'gravity': (fx_gravity, fy_gravity),
    'spring': (fx_linear_spring, fy_linear_spring),
    'cubic': (fx_3, fy_3),
}


def build_solution(data):
    x0 = float(data.get('x0', 0))
    y0 = float(data.get('y0', 0))
    vx0 = float(data.get('vx', 5))
    vy0 = float(data.get('vy', 10))
    dt = float(data.get('dt', 0.05))
    steps = int(data.get('steps', 100))
    model = data.get('model', 'gravity')
    G = float(data.get('G', 1.0))
    M = float(data.get('M', 1.0))
    k = float(data.get('k', 0.0))
    c = float(data.get('c', 0.0))
    fx_expr = (data.get('fx_expr') or '').strip()
    fy_expr = (data.get('fy_expr') or '').strip()

    if model == 'custom':
        if not fx_expr or not fy_expr:
            raise ValueError('カスタムモデルではfx_exprとfy_exprの両方を入力してください。')
        fx_func = make_force_function(fx_expr)
        fy_func = make_force_function(fy_expr)
    elif fx_expr or fy_expr:
        fx_func = make_force_function(fx_expr or '0')
        fy_func = make_force_function(fy_expr or '0')
    else:
        fx_func, fy_func = MODEL_FUNCS.get(model, MODEL_FUNCS['gravity'])

    params = Params(m=1.0, M=M, G=G, k=k, c=c, fx=fx_func, fy=fy_func)
    initials = Initials(x0=x0, y0=y0, vx0=vx0, vy0=vy0)
    t_span = (0.0, dt * steps)

    solution = solve_motion(initials, params, t_span, num_points=steps)
    if not solution.success or len(solution.t) == 0 or solution.y is None or len(solution.y) < 4:
        raise RuntimeError(solution.message if hasattr(solution, 'message') else 'シミュレーションに失敗しました。')

    return solution


def parse_json_request():
    if os.environ.get('REQUEST_METHOD', '').upper() != 'POST':
        raise RuntimeError('POSTのみ対応しています。')
    length = os.environ.get('CONTENT_LENGTH')
    if not length:
        return {}
    data = sys.stdin.read(int(length))
    return json.loads(data)


def respond_excel(content):
    sys.stdout.write('Content-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet\r\n')
    sys.stdout.write('Content-Disposition: attachment; filename="Result.xlsx"\r\n')
    sys.stdout.write('\r\n')
    sys.stdout.buffer.write(content)


if __name__ == '__main__':
    try:
        data = parse_json_request()
        solution = build_solution(data)

        workbook = Workbook()
        summary = workbook.active
        summary.title = 'Summary'
        summary.append(['項目', '値'])
        summary.append(['x0', data.get('x0', '')])
        summary.append(['y0', data.get('y0', '')])
        summary.append(['vx0', data.get('vx', '')])
        summary.append(['vy0', data.get('vy', '')])
        summary.append(['dt', data.get('dt', '')])
        summary.append(['steps', data.get('steps', '')])
        summary.append(['model', data.get('model', '')])
        summary.append(['function', data.get('fx_expr', '') + ' / ' + data.get('fy_expr', '')])

        r = [float(x) for x in solution.y[0]]
        s = [float(y) for y in solution.y[1]]
        radii = [((xi ** 2 + yi ** 2) ** 0.5) for xi, yi in zip(r, s)]
        r_max = max(radii)
        r_min = min(radii)
        a = (r_max + r_min) / 2
        period = None
        dt = float(data.get('dt', 0.05))
        min_time = dt * 5
        x0 = float(data.get('x0', 0))
        y0 = float(data.get('y0', 0))
        tol = max(1e-3, ((x0 ** 2 + y0 ** 2) ** 0.5) * 0.02)
        for t, x, y in zip(solution.t, solution.y[0], solution.y[1]):
            if t > min_time and ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 <= tol:
                period = t
                break

        summary.append(['long axis a', a])
        summary.append(['period T', period if period is not None else '未検出'])
        if period is not None and period > 0:
            summary.append(['a^3/T^2', a ** 3 / (period ** 2)])

        trajectory_ws = workbook.create_sheet('Trajectory')
        trajectory_ws.append(['t', 'x', 'y', 'vx', 'vy'])
        for t, x, y, vx, vy in zip(solution.t, solution.y[0], solution.y[1], solution.y[2], solution.y[3]):
            trajectory_ws.append([float(t), float(x), float(y), float(vx), float(vy)])

        output = BytesIO()
        workbook.save(output)
        respond_excel(output.getvalue())
    except Exception as exc:
        sys.stdout.write('Content-Type: application/json\r\n')
        sys.stdout.write('Status: 500\r\n')
        sys.stdout.write('\r\n')
        sys.stdout.write(json.dumps({'error': str(exc)}, ensure_ascii=False))
