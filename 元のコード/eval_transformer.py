import cv2
import numpy as np
import mediapipe as mp
import torch
import argparse
import math
from collections import Counter
from TransformerModel import TimeSeriesTransformer

def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

def calculate_angle_3d(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm == 0 or v2_norm == 0:
        return 0
    dot_product = np.dot(v1, v2)
    cos_angle = dot_product / (v1_norm * v2_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = math.acos(cos_angle)
    return angle_rad

def extract_hand_features(hand_landmarks, prev_landmarks=None, frame_time=1/30):
    finger_tips = [4, 8, 12, 16, 20]
    finger_roots = [1, 5, 9, 13, 17]
    finger_mids = [2, 6, 10, 14, 18]
    wrist = hand_landmarks.landmark[0]
    middle_root = hand_landmarks.landmark[9]
    features = []
    finger_distances = [calculate_distance(wrist, hand_landmarks.landmark[tip]) for tip in finger_tips]
    features.extend(finger_distances)
    finger_angles = [calculate_angle_3d(hand_landmarks.landmark[mid], middle_root, hand_landmarks.landmark[tip]) for mid, tip in zip(finger_mids, finger_tips)]
    features.extend(finger_angles)
    wrist_middle_distance = calculate_distance(wrist, middle_root)
    features.append(wrist_middle_distance)
    total_distance = sum(finger_distances) / 5
    features.append(total_distance)
    total_angle = sum(finger_angles) / 5
    features.append(total_angle)
    finger_energies = [calculate_distance(hand_landmarks.landmark[tip], hand_landmarks.landmark[root])**2 for tip, root in zip(finger_tips, finger_roots)]
    features.extend(finger_energies)
    finger_momentums = []
    if prev_landmarks is not None:
        for tip in finger_tips:
            current_pos = hand_landmarks.landmark[tip]
            prev_pos = prev_landmarks.landmark[tip]
            x_momentum = (current_pos.x - prev_pos.x) / frame_time
            y_momentum = (current_pos.y - prev_pos.y) / frame_time
            z_momentum = (current_pos.z - prev_pos.z) / frame_time
            finger_momentums.extend([x_momentum, y_momentum, z_momentum])
    else:
        finger_momentums = [0] * 15
    features.extend(finger_momentums)
    if prev_landmarks is not None:
        dx = hand_landmarks.landmark[0].x - prev_landmarks.landmark[0].x
        if dx > 0:
            right_movement = dx
            left_movement = 0
        else:
            right_movement = 0
            left_movement = -dx
    else:
        right_movement = 0
        left_movement = 0
    left_right_ratio = right_movement / (left_movement + 1e-6)
    features.append(left_right_ratio)
    thumb_tip = hand_landmarks.landmark[4]
    thumb_root = hand_landmarks.landmark[1]
    thumb_distance = calculate_distance(thumb_tip, thumb_root)
    if prev_landmarks is not None:
        prev_thumb_tip = prev_landmarks.landmark[4]
        prev_thumb_root = prev_landmarks.landmark[1]
        prev_thumb_distance = calculate_distance(prev_thumb_tip, prev_thumb_root)
    else:
        prev_thumb_distance = thumb_distance
    flexion_extension_ratio = (thumb_distance + 1e-6) / (prev_thumb_distance + 1e-6)
    features.append(flexion_extension_ratio)
    base_lm = hand_landmarks.landmark[0]
    for lm in hand_landmarks.landmark:
        rel_x = lm.x - base_lm.x
        rel_y = lm.y - base_lm.y
        rel_z = lm.z - base_lm.z
        features.extend([rel_x, rel_y, rel_z])
    adjacent_distances = []
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    index_mcp = hand_landmarks.landmark[5]
    middle_mcp = hand_landmarks.landmark[9]
    pinky_mcp = hand_landmarks.landmark[17]
    thumb_ip = hand_landmarks.landmark[3]
    index_pip = hand_landmarks.landmark[7]
    middle_pip = hand_landmarks.landmark[11]
    ring_pip = hand_landmarks.landmark[15]
    pinky_pip = hand_landmarks.landmark[19]
    palm_center = hand_landmarks.landmark[9]
    wrist = hand_landmarks.landmark[0]
    adjacent_distances.extend([
        np.linalg.norm([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y]),
        np.linalg.norm([thumb_tip.x - middle_tip.x, thumb_tip.y - middle_tip.y]),
        np.linalg.norm([thumb_tip.x - ring_tip.x, thumb_tip.y - ring_tip.y]),
        np.linalg.norm([thumb_tip.x - pinky_tip.x, thumb_tip.y - pinky_tip.y]),
        np.linalg.norm([index_tip.x - middle_tip.x, index_tip.y - middle_tip.y]),
        np.linalg.norm([index_tip.x - index_mcp.x, index_tip.y - index_mcp.y]),
        np.linalg.norm([index_pip.x - middle_pip.x, index_pip.y - middle_pip.y]),
        np.linalg.norm([middle_tip.x - palm_center.x, middle_tip.y - palm_center.y]),
        np.linalg.norm([ring_tip.x - pinky_tip.x, ring_tip.y - pinky_tip.y])
    ])
    features.extend(adjacent_distances)
    def calc_angle(base, tip):
        dx = tip.x - base.x
        dy = tip.y - base.y
        return np.arctan2(dy, dx) * (180.0 / np.pi)
    def calc_angle_at_vertex(a, b, c):
        ab = np.array([b.x - a.x, b.y - a.y])
        ac = np.array([c.x - a.x, c.y - a.y])
        dot = np.dot(ab, ac)
        norm_ab = np.linalg.norm(ab)
        norm_ac = np.linalg.norm(ac)
        cos_angle = dot / (norm_ab * norm_ac)
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return angle_rad * (180.0 / np.pi)
    thumb_angle_from_palm = calc_angle(palm_center, thumb_tip)
    index_angle_from_palm = calc_angle(palm_center, index_tip)
    middle_angle_from_palm = calc_angle(palm_center, middle_tip)
    ring_angle_from_palm = calc_angle(palm_center, ring_tip)
    pinky_angle_from_palm = calc_angle(palm_center, pinky_tip)
    thumb_angle_at_vertex = calc_angle_at_vertex(thumb_tip, hand_landmarks.landmark[2], pinky_mcp)
    index_angle_at_vertex = calc_angle_at_vertex(index_tip, hand_landmarks.landmark[2], pinky_mcp)
    middle_angle_at_vertex = calc_angle_at_vertex(middle_tip, hand_landmarks.landmark[2], pinky_mcp)
    ring_angle_at_vertex = calc_angle_at_vertex(ring_tip, hand_landmarks.landmark[2], pinky_mcp)
    pinky_angle_at_vertex = calc_angle_at_vertex(pinky_tip, hand_landmarks.landmark[2], pinky_mcp)
    thumb_angle_from_base = calc_angle(thumb_ip, thumb_tip)
    index_angle_from_base = calc_angle(index_pip, index_tip)
    middle_angle_from_base = calc_angle(middle_pip, middle_tip)
    ring_angle_from_base = calc_angle(ring_pip, ring_tip)
    pinky_angle_from_base = calc_angle(pinky_pip, pinky_tip)
    middleMCP_angle_from_base = calc_angle(wrist, middle_mcp)
    def calc_xz_angle(base, tip):
        dx = tip.x - base.x
        dz = tip.z - base.z
        return np.arctan2(dz, dx) * (180.0 / np.pi)
    thumb_xz_angle = calc_xz_angle(middle_mcp, hand_landmarks.landmark[2])
    pinky_xz_angle = calc_xz_angle(middle_mcp, pinky_pip)
    features.extend([
        thumb_angle_from_palm, index_angle_from_palm, middle_angle_from_palm, ring_angle_from_palm, pinky_angle_from_palm,
        thumb_angle_at_vertex, index_angle_at_vertex, middle_angle_at_vertex, ring_angle_at_vertex, pinky_angle_at_vertex,
        thumb_angle_from_base, index_angle_from_base, middle_angle_from_base, ring_angle_from_base, pinky_angle_from_base,
        middleMCP_angle_from_base, thumb_xz_angle, pinky_xz_angle
    ])
    # 必ず126次元にパディング
    if len(features) < 125:
        features.extend([0] * (125 - len(features)))
    elif len(features) > 125:
        features = features[:125]
    # print(f'特徴量長: {len(features)}')  # デバッグ用
    return features

def process_video(video_path, args, model, device, window=5, save_output=True):
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(video_path)
    prev_landmarks = None
    features = []
    frames = []
    # 動画情報取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30  # デフォルト
    # 出力動画パス
    out_path = video_path.rsplit('.', 1)[0] + '_labeled.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height)) if save_output else None
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                feat = extract_hand_features(hand_landmarks, prev_landmarks)
                prev_landmarks = hand_landmarks
            else:
                feat = [0] * args.feature_size
            features.append(feat)
            frames.append(frame)
    cap.release()
    pred_labels = []
    label_map = {0: 'Static', 1: 'Move', 2: 'Between'}
    with torch.no_grad():
        for i in range(0, len(features) - window + 1, window):
            window_feats = features[i:i+window]
            data = torch.tensor(window_feats, dtype=torch.float).unsqueeze(0)
            output = model(data.to(device))
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_label = int(probs.argmax())
            pred_labels.extend([pred_label]*window)
    if len(pred_labels) < len(frames):
        pred_labels.extend([pred_labels[-1]] * (len(frames) - len(pred_labels)))
    # 動画再生しながらラベルを表示＆保存
    for idx, (frame, label) in enumerate(zip(frames, pred_labels)):
        disp_label = label_map.get(label, str(label))
        cv2.putText(frame, f'Label: {disp_label}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        if out_writer:
            out_writer.write(frame)
        cv2.imshow('Video with Label', frame)
        key = cv2.waitKey(30)
        if key == 27:
            break
    if out_writer:
        out_writer.release()
        print(f'ラベル付き動画を保存しました: {out_path}')
    cv2.destroyAllWindows()
    # 予測ラベルの集計も表示（日本語で表示）
    if pred_labels:
        from collections import Counter
        jp_label_map = {0: '静止指文字', 1: '動作状態', 2: '文字区間'}
        most_common = Counter(pred_labels).most_common(1)[0][0]
        most_common_label = jp_label_map.get(most_common, str(most_common))
        pred_labels_str = [jp_label_map.get(l, str(l)) for l in pred_labels]
        print(f'動画: {video_path}')
        print(f'5フレーム単位の最頻クラス: {most_common_label}（{most_common}）')
        print(f'全予測ラベル: {pred_labels_str}')
    else:
        print(f'{video_path}: 十分なフレームがありません')


def main():
    # ここで直接パスを指定
    video_dir = "../Video/単語分割/指文字単語学習"  # 必要に応じてパスを変更
    model_path = './best_model.pth'
    feature_size = 125
    nhead = 5
    num_classes = 3  # 3クラスに戻す
    window = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformer(
        feature_size=feature_size,
        num_classes=num_classes,
        seq_length=window,
        nhead=nhead
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    import os
    exts = ('.mp4', '.avi', '.mov', '.MOV')
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(exts)]
    if not video_files:
        print('動画ファイルが見つかりません')
        return
    for video_path in sorted(video_files):
        class Args: pass
        args = Args()
        args.feature_size = feature_size
        process_video(video_path, args, model, device, window)

if __name__ == '__main__':
    main()
