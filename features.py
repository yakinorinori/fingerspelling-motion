"""
特徴量の計算（MediaPipe の結果を保存したランドマーク配列から計算する）

  base   : 既存の125次元（Fingerspelling-word-division の extract_hand_features と同じ計算）
  motionA: 案Aの動き特徴量 18次元（手のひら中心の移動方向を窓でまとめたもの）

特徴量セット:
  "base"    → 125次元
  "base+A"  → 143次元

ランドマーク配列: lm (T, 21, 3)。MediaPipe の正規化座標（x/幅, y/高さ, z）。手が無いフレームは NaN。
"""
import math
from types import SimpleNamespace

import numpy as np

FEATURE_SETS = ["base", "base+A"]
BASE_DIM = 125
MOTION_A_NAMES = (
    [f"dir8_{d}" for d in ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]]  # A1 8方向ヒストグラム
    + ["hv_ratio"]                                                       # A2 水平・垂直の比
    + ["consistency_x", "consistency_y"]                                 # A3 向きの一貫性
    + ["straightness"]                                                   # A4 まっすぐ度
    + ["pca_cos2", "pca_sin2", "linearity"]                             # A5 主方向と直線らしさ
    + ["rotation"]                                                       # A6 回転
    + ["scale_change"]                                                   # A7 手前・奥（手の大きさの変化）
    + ["path_length"]                                                    # 窓内の移動量
)
MOTION_A_DIM = len(MOTION_A_NAMES)  # 18


# ---------------------------------------------------------------------------
# 既存の125次元（eval_transformer.py の extract_hand_features をそのまま移植）
# 違いは「前フレーム」を動画ごとに渡すこと（元はグローバル変数で動画をまたいでいた）
# ---------------------------------------------------------------------------
def _calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)


def _calculate_angle_3d(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0
    return math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))


def _extract_hand_features(hand_landmarks, prev_landmarks=None, frame_time=1 / 30):
    L = hand_landmarks.landmark
    finger_tips = [4, 8, 12, 16, 20]
    finger_roots = [1, 5, 9, 13, 17]
    finger_mids = [2, 6, 10, 14, 18]
    wrist, middle_root = L[0], L[9]
    f = []
    finger_distances = [_calculate_distance(wrist, L[t]) for t in finger_tips]
    f.extend(finger_distances)
    finger_angles = [_calculate_angle_3d(L[m], middle_root, L[t]) for m, t in zip(finger_mids, finger_tips)]
    f.extend(finger_angles)
    f.append(_calculate_distance(wrist, middle_root))
    f.append(sum(finger_distances) / 5)
    f.append(sum(finger_angles) / 5)
    f.extend([_calculate_distance(L[t], L[r]) ** 2 for t, r in zip(finger_tips, finger_roots)])
    if prev_landmarks is not None:
        P = prev_landmarks.landmark
        for t in finger_tips:
            f.extend([(L[t].x - P[t].x) / frame_time, (L[t].y - P[t].y) / frame_time, (L[t].z - P[t].z) / frame_time])
        dx = L[0].x - P[0].x
        right, left = (dx, 0) if dx > 0 else (0, -dx)
    else:
        f.extend([0] * 15)
        right = left = 0
    f.append(right / (left + 1e-6))
    thumb_distance = _calculate_distance(L[4], L[1])
    prev_thumb = _calculate_distance(prev_landmarks.landmark[4], prev_landmarks.landmark[1]) \
        if prev_landmarks is not None else thumb_distance
    f.append((thumb_distance + 1e-6) / (prev_thumb + 1e-6))
    base = L[0]
    for lm in L:
        f.extend([lm.x - base.x, lm.y - base.y, lm.z - base.z])

    thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip = L[4], L[8], L[12], L[16], L[20]
    index_mcp, middle_mcp, pinky_mcp = L[5], L[9], L[17]
    thumb_ip, index_pip, middle_pip, ring_pip, pinky_pip = L[3], L[7], L[11], L[15], L[19]
    palm_center = L[9]
    d2 = lambda a, b: np.linalg.norm([a.x - b.x, a.y - b.y])
    f.extend([d2(thumb_tip, index_tip), d2(thumb_tip, middle_tip), d2(thumb_tip, ring_tip),
              d2(thumb_tip, pinky_tip), d2(index_tip, middle_tip), d2(index_tip, index_mcp),
              d2(index_pip, middle_pip), d2(middle_tip, palm_center), d2(ring_tip, pinky_tip)])

    def calc_angle(b, t):
        return np.arctan2(t.y - b.y, t.x - b.x) * (180.0 / np.pi)

    def calc_angle_at_vertex(a, b, c):
        ab = np.array([b.x - a.x, b.y - a.y])
        ac = np.array([c.x - a.x, c.y - a.y])
        cos = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac) + 1e-12)
        return np.arccos(np.clip(cos, -1.0, 1.0)) * (180.0 / np.pi)

    def calc_xz_angle(b, t):
        return np.arctan2(t.z - b.z, t.x - b.x) * (180.0 / np.pi)

    tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
    f.extend([calc_angle(palm_center, t) for t in tips])
    f.extend([calc_angle_at_vertex(t, L[2], pinky_mcp) for t in tips])
    f.extend([calc_angle(b, t) for b, t in zip([thumb_ip, index_pip, middle_pip, ring_pip, pinky_pip], tips)])
    f.append(calc_angle(wrist, middle_mcp))
    f.append(calc_xz_angle(middle_mcp, L[2]))
    f.append(calc_xz_angle(middle_mcp, pinky_pip))
    return f[:BASE_DIM] + [0] * (BASE_DIM - len(f))


def _as_landmarks(arr):
    return SimpleNamespace(landmark=[SimpleNamespace(x=float(a[0]), y=float(a[1]), z=float(a[2])) for a in arr])


def base_features(lm, fps=30.0):
    """(T,21,3) → (T,125)。手が無いフレームは0（既存と同じ）"""
    T = len(lm)
    out = np.zeros((T, BASE_DIM), dtype=np.float32)
    prev = None
    for t in range(T):
        if np.isnan(lm[t]).any():
            continue
        cur = _as_landmarks(lm[t])
        out[t] = _extract_hand_features(cur, prev, frame_time=1 / 30)
        prev = cur
    return out


# ---------------------------------------------------------------------------
# 案A：手のひら中心の動きを窓でまとめた特徴量
# ---------------------------------------------------------------------------
PALM_IDS = [0, 5, 9, 13, 17]


def _interp(values, valid):
    """手が無いフレームを前後から線形補間（端は最も近い値で埋める）"""
    t = np.arange(len(values))
    out = np.empty_like(values)
    for d in range(values.shape[1]):
        out[:, d] = np.interp(t, t[valid], values[valid, d])
    return out


def motion_a_features(lm, width, height, half_window=7):
    """(T,21,3) → (T,18)。窓は前後 half_window フレーム（初期値7 = 約0.5秒 @30fps）"""
    T = len(lm)
    out = np.zeros((T, MOTION_A_DIM), dtype=np.float32)
    valid = ~np.isnan(lm).any(axis=(1, 2))
    if valid.sum() < 2:
        return out

    # 画素単位に直す（正規化座標のままだと、横長の動画で横の動きが縮んで見える）
    xy = lm[:, :, :2] * np.array([width, height], dtype=np.float64)
    center = _interp(np.nanmean(xy[:, PALM_IDS], axis=1), valid)       # 手のひら中心 (T,2)
    hand_vec = _interp(xy[:, 9] - xy[:, 0], valid)                      # 手首→中指の付け根 (T,2)
    size = np.linalg.norm(hand_vec, axis=1)                             # 手の大きさ
    scale = float(np.median(size[valid])) + 1e-6

    pos = center / scale                                                # 手の大きさで割った位置
    vel = np.vstack([[0, 0], np.diff(pos, axis=0)])                     # (T,2)
    vel[:, 1] *= -1                                                     # 画像の y は下向きなので、上を正にする
    speed = np.linalg.norm(vel, axis=1)
    theta = np.arctan2(vel[:, 1], vel[:, 0])
    bins = (np.floor((theta + np.pi / 8) / (np.pi / 4)).astype(int)) % 8
    angle = np.unwrap(np.arctan2(hand_vec[:, 1], hand_vec[:, 0]))
    d_angle = np.concatenate([[0], np.diff(angle)])
    d_scale = np.concatenate([[0], np.diff(np.log(size + 1e-6))])

    eps = 1e-8
    for t in range(T):
        a, b = max(0, t - half_window), min(T, t + half_window + 1)
        v, s = vel[a:b], speed[a:b]
        hist = np.bincount(bins[a:b], weights=s, minlength=8)
        path = s.sum()
        if path > eps:
            sx2, sy2 = (v[:, 0] ** 2).sum(), (v[:, 1] ** 2).sum()
            hv = sx2 / (sx2 + sy2 + eps)
            cons_x = v[:, 0].sum() / (np.abs(v[:, 0]).sum() + eps)
            cons_y = v[:, 1].sum() / (np.abs(v[:, 1]).sum() + eps)
            straight = np.linalg.norm(v.sum(axis=0)) / (path + eps)
        else:
            hv, cons_x, cons_y, straight = 0.5, 0.0, 0.0, 0.0
        p = pos[a:b] * np.array([1, -1])
        cov = np.cov(p.T) if len(p) > 1 else np.zeros((2, 2))
        w, vecs = np.linalg.eigh(cov)
        if w.sum() > eps:
            pc = vecs[:, 1]
            phi = math.atan2(pc[1], pc[0])
            cos2, sin2, lin = math.cos(2 * phi), math.sin(2 * phi), w[1] / (w.sum() + eps)
        else:
            cos2 = sin2 = lin = 0.0
        out[t] = [*hist, hv, cons_x, cons_y, straight, cos2, sin2, lin,
                  d_angle[a:b].sum(), d_scale[a:b].sum(), path]
    return out


def compute_features(lm, width, height, feature_set, fps=30.0):
    base = base_features(lm, fps)
    if feature_set == "base":
        return base
    if feature_set == "base+A":
        return np.concatenate([base, motion_a_features(lm, width, height)], axis=1)
    raise ValueError(f"未知の特徴量セット: {feature_set}")
