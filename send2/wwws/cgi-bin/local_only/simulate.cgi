#!/usr/bin/env python3
import json
import os
import sys

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


def respond_json(payload, status=200):
    sys.stdout.write('Content-Type: application/json\r\n')
    sys.stdout.write(f'Status: {status}\r\n')
    sys.stdout.write('\r\n')
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))


if __name__ == '__main__':
    try:
        data = parse_json_request()
        solution = build_solution(data)
        points = [
            {
                't': float(t),
                'x': float(x),
                'y': float(y),
                'vx': float(vx),
                'vy': float(vy),
            }
            for t, x, y, vx, vy in zip(
                solution.t,
                solution.y[0],
                solution.y[1],
                solution.y[2],
                solution.y[3],
            )
        ]
        respond_json(points)
    except Exception as exc:
        respond_json({'error': str(exc)}, status=500)
