"""Result.xlsx를 과제 설명서의 평가 기준으로 다시 채점한다.

사용법:
    python evaluation/evaluate.py data.json distance-data.txt Result.xlsx [--json out.json]

과제 설명서 기준
- 라우팅 비용 = 차량 수 x 150,000원 + 총 이동 거리(km) x 500원
- 하차 비용 = 셔플링 횟수 x 500원
- 적재함 160(X) x 280(Y) x 180(Z) cm, 원점은 가장 안쪽 좌측, 문은 Y=280 평면, 후입선출
- 각 착지는 차량 한 대가 한 번만 방문

셔플링 횟수의 세부 판정은 설명서에 "상품을 꺼내기 위하여 이동해야 하는 주변 상품의 수"로만
나와 있어 아래처럼 정했다 (공식 채점기와 다를 수 있음).
- 착지 순서대로 하차한다. 착지 k에서 내릴 박스를 꺼낼 때, 아직 실려 있는 다른 착지 박스 중
  그 박스의 위(XY가 겹치고 더 높은 곳)나 문 쪽 앞(XZ가 겹치고 Y가 더 큰 곳)에 있는 박스를 치운다.
- 치워야 하는 박스를 꺼내려면 그 박스를 막는 박스도 치워야 하므로 막는 관계를 끝까지 따라간다.
- 한 착지에서 치운 박스는 한 번만 센다 (꺼냈다가 다시 넣는 것을 1회로 본다).
"""
import argparse
import json
import sys
from collections import defaultdict

import pandas as pd

TRUCK_W, TRUCK_L, TRUCK_H = 160, 280, 180
FIXED_COST = 150_000
FUEL_COST_PER_KM = 500
SHUFFLE_COST = 500
EPS = 1e-6


def load_distance(path):
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


def get_dist(dist, a, b):
    if a == b:
        return 0
    if (a, b) in dist:
        return dist[(a, b)]
    if (b, a) in dist:
        return dist[(b, a)]
    raise KeyError(f"거리 없음: {a} -> {b}")


def overlap_1d(a0, a1, b0, b1):
    return min(a1, b1) - max(a0, b0) > EPS


def blocks(blocker, target):
    """blocker가 target을 꺼낼 때 방해하면 True (위 또는 문 쪽 앞)."""
    above = (overlap_1d(blocker["x"], blocker["x2"], target["x"], target["x2"])
             and overlap_1d(blocker["y"], blocker["y2"], target["y"], target["y2"])
             and blocker["z"] >= target["z2"] - EPS)
    front = (overlap_1d(blocker["x"], blocker["x2"], target["x"], target["x2"])
             and overlap_1d(blocker["z"], blocker["z2"], target["z"], target["z2"])
             and blocker["y"] >= target["y2"] - EPS)
    return above or front


def evaluate(data, dist, df):
    orders = {o["box_id"]: o for o in data["orders"]}
    errors = defaultdict(list)

    boxes_df = df[df["Box_ID"].notna() & (df["Box_ID"].astype(str) != "")]
    vehicles = sorted(df["Vehicle_ID"].unique())

    # 박스 누락·중복
    counts = boxes_df["Box_ID"].value_counts()
    dup = counts[counts > 1]
    if len(dup):
        errors["중복 박스"] = list(dup.index[:10])
    missing = set(orders) - set(boxes_df["Box_ID"])
    if missing:
        errors["누락 박스"] = sorted(missing)[:10] + ([f"... 총 {len(missing)}개"] if len(missing) > 10 else [])

    # 착지 1대 1회 방문
    dest_vehicles = boxes_df.groupby("Destination")["Vehicle_ID"].nunique()
    split = dest_vehicles[dest_vehicles > 1]

    total_dist_m = 0
    total_shuffle = 0
    per_vehicle = []
    floating = 0

    for v in vehicles:
        dv = df[df["Vehicle_ID"] == v].sort_values("Route_Order", kind="stable")
        stops = []
        for d in dv["Destination"].astype(str):
            if not stops or stops[-1] != d:
                stops.append(d)
        if stops[0] != "Depot":
            stops.insert(0, "Depot")
        if stops[-1] != "Depot":
            stops.append("Depot")
        inner = [s for s in stops[1:-1]]
        if len(inner) != len(set(inner)):
            errors["한 차량이 같은 착지 재방문"].append(int(v))
        vdist = sum(get_dist(dist, stops[i], stops[i + 1]) for i in range(len(stops) - 1))
        total_dist_m += vdist

        bv = boxes_df[boxes_df["Vehicle_ID"] == v]
        boxes = []
        for _, r in bv.iterrows():
            b = {
                "id": r["Box_ID"], "dest": str(r["Destination"]),
                "x": float(r["Lower_Left_X"]), "y": float(r["Lower_Left_Y"]), "z": float(r["Lower_Left_Z"]),
                "w": float(r["Box_Width"]), "l": float(r["Box_Length"]), "h": float(r["Box_Height"]),
            }
            b["x2"], b["y2"], b["z2"] = b["x"] + b["w"], b["y"] + b["l"], b["z"] + b["h"]
            o = orders.get(b["id"])
            if o and sorted([b["w"], b["l"], b["h"]]) != sorted(o["dimension"].values()):
                errors["치수 불일치"].append(b["id"])
            if o and o["destination"] != b["dest"]:
                errors["착지 불일치"].append(b["id"])
            if b["x"] < -EPS or b["y"] < -EPS or b["z"] < -EPS or b["x2"] > TRUCK_W + EPS \
                    or b["y2"] > TRUCK_L + EPS or b["z2"] > TRUCK_H + EPS:
                errors["적재함 범위 초과"].append(b["id"])
            boxes.append(b)

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, c = boxes[i], boxes[j]
                if (overlap_1d(a["x"], a["x2"], c["x"], c["x2"]) and overlap_1d(a["y"], a["y2"], c["y"], c["y2"])
                        and overlap_1d(a["z"], a["z2"], c["z"], c["z2"])):
                    errors["박스 겹침"].append(f"{a['id']}-{c['id']}")
        # 바닥이나 다른 박스 위에 놓이지 않은 박스 (설명서 제약은 아니지만 참고용)
        for b in boxes:
            if b["z"] > EPS and not any(
                    abs(o["z2"] - b["z"]) < EPS and overlap_1d(o["x"], o["x2"], b["x"], b["x2"])
                    and overlap_1d(o["y"], o["y2"], b["y"], b["y2"]) for o in boxes):
                floating += 1

        # 하차 시뮬레이션
        remaining = list(boxes)
        vshuffle = 0
        for stop in inner:
            targets = [b for b in remaining if b["dest"] == stop]
            if not targets:
                continue
            others = [b for b in remaining if b["dest"] != stop]
            to_move = set()
            frontier = list(targets)
            while frontier:
                t = frontier.pop()
                for o in others:
                    if o["id"] not in to_move and blocks(o, t):
                        to_move.add(o["id"])
                        frontier.append(o)
            vshuffle += len(to_move)
            remaining = others
        total_shuffle += vshuffle

        per_vehicle.append({
            "vehicle": int(v), "stops": len(inner), "boxes": len(boxes),
            "distance_km": round(vdist / 1000, 3), "shuffle": vshuffle,
            "load_ratio": round(sum(b["w"] * b["l"] * b["h"] for b in boxes) / (TRUCK_W * TRUCK_L * TRUCK_H), 3),
        })

    n_veh = len(vehicles)
    fixed = n_veh * FIXED_COST
    fuel = total_dist_m / 1000 * FUEL_COST_PER_KM
    shuffle_cost = total_shuffle * SHUFFLE_COST
    return {
        "vehicles": n_veh,
        "boxes_loaded": int(boxes_df["Box_ID"].nunique()),
        "boxes_total": len(orders),
        "distance_km": round(total_dist_m / 1000, 3),
        "fixed_cost": fixed,
        "fuel_cost": round(fuel),
        "routing_cost": round(fixed + fuel),
        "shuffle_count": total_shuffle,
        "unloading_cost": shuffle_cost,
        "total_score": round(fixed + fuel + shuffle_cost),
        "split_destinations": int(len(split)),
        "floating_boxes": floating,
        "errors": {k: v[:10] for k, v in errors.items() if v},
        "per_vehicle": per_vehicle,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("distance")
    ap.add_argument("result")
    ap.add_argument("--json")
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    dist = load_distance(args.distance)
    df = pd.read_excel(args.result)
    r = evaluate(data, dist, df)

    print(f"차량 {r['vehicles']}대, 박스 {r['boxes_loaded']}/{r['boxes_total']}, 이동 {r['distance_km']:,} km")
    print(f"라우팅 비용 {r['routing_cost']:,}원 (고정 {r['fixed_cost']:,} + 유류 {r['fuel_cost']:,})")
    print(f"하차 비용 {r['unloading_cost']:,}원 (셔플링 {r['shuffle_count']}회)")
    print(f"TOTAL SCORE {r['total_score']:,}원")
    print(f"여러 차량에 나뉜 착지 {r['split_destinations']}곳 (규칙: 착지당 차량 1대)")
    print(f"공중에 뜬 박스 {r['floating_boxes']}개 (참고)")
    if r["errors"]:
        print("오류:", json.dumps(r["errors"], ensure_ascii=False))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
