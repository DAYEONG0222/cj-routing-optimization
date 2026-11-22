"""AI 기반 배송 라우팅 및 적재 순서 최적화.

실행: python main.py data.json distance-data.txt  →  Result.xlsx

흐름
1. 착지 단위 CVRP: 착지마다 주문 부피를 합쳐 수요로 두고, OR-Tools로 차량 배정과 방문 순서를 함께 푼다.
   착지는 차량 한 대에만 배정된다 (과제 규칙).
2. 배송 역순 적재: 마지막 착지 박스부터 적재함 안쪽(Y=0)에 싣는다. 이미 실린 '나중 착지' 박스가
   새 박스의 위나 문 쪽 앞을 막는 위치는 후보에서 뺀다 → 하차할 때 셔플링이 생기지 않는다.
3. 적재 실패 처리: 경로상 부피는 맞아도 3D로 안 들어가는 착지는 다른 차량에 끼워 넣고,
   그래도 남으면 새 차량을 쓴다.
"""
import json
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

TRUCK_W, TRUCK_L, TRUCK_H = 160, 280, 180          # X(넓이), Y(깊이, 문은 Y=280), Z(높이)
TRUCK_VOL = TRUCK_W * TRUCK_L * TRUCK_H
FIXED_COST = 150_000                               # 원/대
FUEL_COST_PER_M = 0.5                              # 500원/km
SHUFFLE_COST = 500                                 # 원/회
SUPPORT_RATIO = 0.6                                # 박스 바닥 면적 중 받쳐져야 하는 비율

VRP_FILL = 0.80                                    # CVRP 용량 = 적재함 부피 x 이 비율 (0.72~0.88 비교 후 선택)
VRP_TIME_LIMIT = 60                                # 초
TIME_BUDGET = 270                                  # 전체 목표 시간(초). 넘으면 보조 탐색을 줄인다

START = time.time()


# ---------------------------------------------------------------- 데이터
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_distance(path):
    """방향이 있는 거리(m). 제공 OD 행렬은 A→B와 B→A가 다르다."""
    dist = {}
    with open(path, encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            o, d, _, m = parts
            dist[(o, d)] = int(float(m))
    return dist


def dget(dist, a, b):
    if a == b:
        return 0
    return dist.get((a, b), dist.get((b, a), 10 ** 7))


# ---------------------------------------------------------------- 1. 라우팅
def solve_cvrp(dests, demand, dist, fill, time_limit):
    """dests: 착지 ID 목록. 반환: 차량별 착지 순서 리스트."""
    nodes = ["Depot"] + dests
    n = len(nodes)
    cap = int(TRUCK_VOL * fill)
    total = sum(demand[d] for d in dests)
    n_veh = max(1, -(-total // cap)) + 3

    manager = pywrapcp.RoutingIndexManager(n, n_veh, 0)
    routing = pywrapcp.RoutingModel(manager)
    # 비용 단위: 원 x 2 (거리 1m = 0.5원 → 1, 고정비 150,000원 → 300,000)
    matrix = [[dget(dist, a, b) for b in nodes] for a in nodes]

    def arc(i, j):
        return matrix[manager.IndexToNode(i)][manager.IndexToNode(j)]

    cb = routing.RegisterTransitCallback(arc)
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    routing.SetFixedCostOfAllVehicles(int(FIXED_COST / FUEL_COST_PER_M))

    dem = [0] + [demand[d] for d in dests]
    dcb = routing.RegisterUnaryTransitCallback(lambda i: dem[manager.IndexToNode(i)])
    routing.AddDimensionWithVehicleCapacity(dcb, 0, [cap] * n_veh, True, "Volume")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = int(time_limit)
    sol = routing.SolveWithParameters(params)
    if sol is None:
        raise RuntimeError("CVRP 해를 찾지 못함")

    routes = []
    for v in range(n_veh):
        idx = routing.Start(v)
        r = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node:
                r.append(nodes[node])
            idx = sol.Value(routing.NextVar(idx))
        if r:
            routes.append(r)
    return routes


def route_distance(route, dist):
    stops = ["Depot"] + route + ["Depot"]
    return sum(dget(dist, stops[i], stops[i + 1]) for i in range(len(stops) - 1))


def best_insertion(route, d, dist):
    """착지 d를 route에 넣을 때 거리가 가장 적게 늘어나는 위치와 증가량."""
    stops = ["Depot"] + route + ["Depot"]
    best = (None, float("inf"))
    for i in range(len(stops) - 1):
        delta = dget(dist, stops[i], d) + dget(dist, d, stops[i + 1]) - dget(dist, stops[i], stops[i + 1])
        if delta < best[1]:
            best = (i, delta)
    return best


# ---------------------------------------------------------------- 2. 적재
def _sliver(g):
    return 1 if 0 < g < 3 else 0   # 30cm 미만 틈은 어떤 박스도 못 쓴다


# 후보 위치 우선순위(작을수록 먼저, 단위는 격자 칸). 경로마다 여러 규칙으로 실어 보고 가장 잘 들어간 것을 쓴다.
POSITION_KEYS = [
    lambda x, y, z, w, l, h: (y + l, z, x),                                   # 앞면이 안쪽인 곳 → 아래 → 왼쪽
    lambda x, y, z, w, l, h: (y + l + 2 * (_sliver(x) + _sliver(GX - x - w)), z, x),  # + 벽 쪽 틈 벌점
    lambda x, y, z, w, l, h: (y + z, y, x),                                   # 안쪽과 아래를 함께
    lambda x, y, z, w, l, h: (y, z, x),                                       # 안쪽 → 아래 → 왼쪽
    lambda x, y, z, w, l, h: (y + l, z, -x),                                  # 오른쪽부터
]
position_key = POSITION_KEYS[0]


CELL = 10                                         # 격자 한 칸 = 10cm (박스·적재함 치수가 모두 10의 배수)
GX, GY, GZ = TRUCK_W // CELL, TRUCK_L // CELL, TRUCK_H // CELL


class Packer:
    """배송 역순으로 박스를 넣는 10cm 격자 높이맵 휴리스틱.

    - 박스는 각 칸의 현재 높이 위에만 놓인다(아래가 빈 채로 뜨지 않음). 바닥 면적의 SUPPORT_RATIO 이상이 받쳐져야 한다.
    - rank: 배송 순서(0이 첫 착지). 이미 실린 박스는 모두 rank가 같거나 더 큰(나중) 착지다.
    - 높이맵 위에 쌓으므로 새 박스 '위'에는 아무것도 없다. 남는 위험은 새 박스의 문 쪽 앞(Y가 더 큰 곳)에
      나중 착지 박스가 있는 경우뿐이고, strict=True면 그런 위치를 뺀다.
    - 후보 위치는 (안쪽 Y, 아래 Z, 왼쪽 X) 순으로 보고 처음 되는 곳에 놓는다.
    """

    def __init__(self):
        self.hmap = np.zeros((GX, GY), dtype=np.int16)
        self.rank = np.full((GX, GY, GZ), -1, dtype=np.int32)   # 칸을 차지한 박스의 rank
        self.boxes = []

    def copy(self):
        p = Packer.__new__(Packer)
        p.hmap, p.rank, p.boxes = self.hmap.copy(), self.rank.copy(), list(self.boxes)
        return p

    def _candidates(self, w, l, h):
        """방향 (w, l, h)로 놓을 수 있는 (y, z, x) 후보. 높이·받침 조건만 본 상태."""
        win = sliding_window_view(self.hmap, (w, l))             # (GX-w+1, GY-l+1, w, l)
        top = win.max(axis=(2, 3))
        support = (win == top[:, :, None, None]).sum(axis=(2, 3))
        ok = (top + h <= GZ) & ((top == 0) | (support >= SUPPORT_RATIO * w * l))
        xs, ys = np.nonzero(ok)
        zs = top[xs, ys]
        return sorted((position_key(x, y, z, w, l, h), x, y, z)
                      for x, y, z in zip(xs.tolist(), ys.tolist(), zs.tolist()))

    def _blocked(self, x, y, z, w, l, h, rank):
        front = self.rank[x:x + w, y + l:, z:z + h]
        return bool((front > rank).any())

    def place(self, box, rank, strict=True):
        dims = box["dimension"]
        w0, l0, h0 = dims["width"] // CELL, dims["length"] // CELL, dims["height"] // CELL
        free, blocked = None, None                                 # 막히지 않는 최선 / 막히는 최선
        for (w, l, h) in sorted({(w0, l0, h0), (l0, w0, h0)}):     # 세운 채로 90도 회전만 허용
            if w > GX or l > GY:
                continue
            for (key, x, y, z) in self._candidates(w, l, h):
                if free is not None and key >= free[0]:
                    break
                if not self._blocked(x, y, z, w, l, h, rank):
                    free = (key, (x, y, z, w, l, h))
                    break
                if not strict and (blocked is None or key < blocked[0]):
                    blocked = (key, (x, y, z, w, l, h))
        best = free or blocked                                     # 막히는 위치는 strict=False일 때만
        if best is None:
            return None
        x, y, z, w, l, h = best[1]
        self.hmap[x:x + w, y:y + l] = z + h
        self.rank[x:x + w, y:y + l, z:z + h] = rank
        b = {"x": x * CELL, "y": y * CELL, "z": z * CELL, "w": w * CELL, "l": l * CELL, "h": h * CELL,
             "rank": rank, "order": box}
        self.boxes.append(b)
        return b


BOX_ORDERS = [
    lambda o: -o["volume"],                                                   # 큰 박스부터
    lambda o: o["volume"],                                                    # 작은 박스부터
]


def pack_route(route, orders_by_dest, strict=True):
    """여러 규칙(위치 우선순위 x 착지 안 박스 순서)으로 실어 보고, 못 실은 부피가 가장 작은 결과를 고른다."""
    global position_key
    best = None
    for key in POSITION_KEYS:
        for box_order in BOX_ORDERS:
            position_key = key
            boxes, failed = _pack_once(route, orders_by_dest, strict, box_order)
            lost = sum(o["volume"] for d in failed for o in orders_by_dest[d])
            if best is None or lost < best[0]:
                best = (lost, boxes, failed)
            if lost == 0:
                position_key = POSITION_KEYS[0]
                return boxes, failed
    position_key = POSITION_KEYS[0]
    return best[1], best[2]


def _pack_once(route, orders_by_dest, strict, box_order):
    """route 순서대로 배송할 때의 적재. 실패한 착지(일부 박스라도 못 실은 착지)를 함께 돌려준다."""
    packer = Packer()
    failed = []
    for rank in range(len(route) - 1, -1, -1):           # 마지막 착지부터
        d = route[rank]
        boxes = sorted(orders_by_dest[d], key=box_order)
        snapshot = packer.copy()
        ok = True
        for o in boxes:
            if packer.place(o, rank, strict) is None:
                ok = False
                break
        if not ok:                                       # 착지 단위로 되돌림 (착지는 한 차량에만)
            packer = snapshot
            failed.append(d)
    return packer.boxes, failed


# ---------------------------------------------------------------- 3. 계획 전체
def plan(data, dist):
    orders_by_dest = defaultdict(list)
    for o in data["orders"]:
        dm = o["dimension"]
        o["volume"] = dm["width"] * dm["length"] * dm["height"]
        orders_by_dest[o["destination"]].append(o)
    dests = sorted(orders_by_dest)
    demand = {d: sum(o["volume"] for o in orders_by_dest[d]) for d in dests}

    routes = solve_cvrp(dests, demand, dist, VRP_FILL, VRP_TIME_LIMIT)

    packed, leftover = {}, []
    for i, r in enumerate(routes):
        boxes, failed = pack_route(r, orders_by_dest, strict=True)
        if failed:
            routes[i] = [d for d in r if d not in failed]
            boxes, failed2 = pack_route(routes[i], orders_by_dest, strict=True)
            leftover += failed + failed2
            routes[i] = [d for d in routes[i] if d not in failed2]
        packed[i] = boxes

    # 남은 착지: 기존 차량에 끼워 넣기(거리 증가가 작은 차량부터) → 안 되면 새 차량
    for d in sorted(leftover, key=lambda d: -demand[d]):
        cands = sorted(((best_insertion(r, d, dist), i) for i, r in enumerate(routes)), key=lambda t: t[0][1])
        done = False
        for (pos, _), i in cands:
            if time.time() - START > TIME_BUDGET:
                break
            trial = routes[i][:pos] + [d] + routes[i][pos:]
            boxes, failed = pack_route(trial, orders_by_dest, strict=True)
            if not failed:
                routes[i], packed[i] = trial, boxes
                done = True
                break
        if not done:
            # 셔플링을 일부 허용하면 들어가는지 (새 차량 15만 원 vs 셔플링 1회 500원)
            for (pos, _), i in cands[:3]:
                trial = routes[i][:pos] + [d] + routes[i][pos:]
                boxes, failed = pack_route(trial, orders_by_dest, strict=False)
                if not failed:
                    routes[i], packed[i] = trial, boxes
                    done = True
                    break
        if not done:
            routes.append([d])
            packed[len(routes) - 1], _ = pack_route([d], orders_by_dest, strict=True)

    out = [(r, packed[i]) for i, r in enumerate(routes) if r]
    return out


# ---------------------------------------------------------------- 출력
def write_result(data, plan_out, path="Result.xlsx"):
    coords = {d["destination_id"]: d["location"] for d in data["destinations"]}
    coords["Depot"] = data["depot"]["location"]
    rows = []

    def depot_row(v, ro):
        return {"Vehicle_ID": v, "Route_Order": ro, "Destination": "Depot", "Order_Number": "", "Box_ID": "",
                "Stacking_Order": "", "Lower_Left_X": 0, "Lower_Left_Y": 0, "Lower_Left_Z": 0,
                "Longitude": coords["Depot"]["longitude"], "Latitude": coords["Depot"]["latitude"],
                "Box_Width": 0, "Box_Length": 0, "Box_Height": 0}

    for v, (route, boxes) in enumerate(plan_out):
        rows.append(depot_row(v, 0))
        stack = {id(b): k + 1 for k, b in enumerate(boxes)}   # 실은 순서 (1이 가장 먼저)
        by_dest = defaultdict(list)
        for b in boxes:
            by_dest[b["order"]["destination"]].append(b)
        for ro, d in enumerate(route, 1):
            for b in sorted(by_dest[d], key=lambda b: stack[id(b)]):
                o = b["order"]
                rows.append({"Vehicle_ID": v, "Route_Order": ro, "Destination": d,
                             "Order_Number": o["order_number"], "Box_ID": o["box_id"],
                             "Stacking_Order": stack[id(b)],
                             "Lower_Left_X": b["x"], "Lower_Left_Y": b["y"], "Lower_Left_Z": b["z"],
                             "Longitude": coords[d]["longitude"], "Latitude": coords[d]["latitude"],
                             "Box_Width": b["w"], "Box_Length": b["l"], "Box_Height": b["h"]})
        rows.append(depot_row(v, len(route) + 1))
    pd.DataFrame(rows).to_excel(path, index=False)


def summarize(data, dist, plan_out):
    n = len(plan_out)
    total_m = sum(route_distance(r, dist) for r, _ in plan_out)
    loaded = sum(len(b) for _, b in plan_out)
    fixed = n * FIXED_COST
    fuel = total_m * FUEL_COST_PER_M
    print(f"처리율: {loaded} / {len(data['orders'])}")
    print(f"차량 수: {n}")
    print(f"고정비용: {fixed:,}원")
    print(f"유류비: {fuel:,.0f}원 ({total_m / 1000:,.1f} km)")
    print(f"실행 소요 시간: {time.time() - START:.1f}초")
    print("셔플링비는 evaluation/evaluate.py로 Result.xlsx를 채점해 확인")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py data.json distance-data.txt")
        sys.exit(1)
    sys.stdout.reconfigure(encoding="utf-8")
    data = load_json(sys.argv[1])
    dist = load_distance(sys.argv[2])
    result = plan(data, dist)
    write_result(data, result)
    summarize(data, dist, result)
