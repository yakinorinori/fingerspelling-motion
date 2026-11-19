# 静止状態，動作状態の動画から特徴量を抽出してcsvに保存するスクリプト
import os
import csv
import cv2
import numpy as np
import mediapipe as mp
import math

mp_hands = mp.solutions.hands

def calculate_distance(point1, point2):
    """2点間の距離計算"""
    return np.sqrt((point1.x - point2.x)**2 + 
                   (point1.y - point2.y)**2 + 
                   (point1.z - point2.z)**2)

def calculate_angle_3d(p1, p2, p3):
    """3Dランドマーク点群から角度を計算"""
    # 3D座標ベクトルの作成
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    
    # ベクトルの長さ
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    # 0除算防止
    if v1_norm == 0 or v2_norm == 0:
        return 0
    
    # 内積
    dot_product = np.dot(v1, v2)
    
    # コサイン類似度
    cos_angle = dot_product / (v1_norm * v2_norm)
    
    # 数値的安定性のためクリッピング
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # 角度に変換（ラジアンから度へ）
    angle_rad = math.acos(cos_angle)
    
    return angle_rad

# グローバル変数として前回のランドマークを保存
prev_landmarks = None

def extract_hand_features(hand_landmarks, frame_time=1/30):  # フレームレート30FPSを仮定
    """手の特徴量抽出（各指先のx,y,z方向の運動量を含む）"""
    global prev_landmarks
    
    # 指定するランドマークインデックス
    finger_tips = [4, 8, 12, 16, 20]  # 親指、人差し指、中指、薬指、小指
    finger_roots = [1, 5, 9, 13, 17]  # 各指の付け根
    finger_mids = [2, 6, 10, 14, 18]  # 各指の中間点
    
    wrist = hand_landmarks.landmark[0]
    middle_root = hand_landmarks.landmark[9]
    
    # 特徴量リスト
    features = []
    
    # 手首(0)基準の指先の相対運動量（距離）: 5個
    finger_distances = [
        calculate_distance(wrist, hand_landmarks.landmark[tip]) 
        for tip in finger_tips
    ]
    features.extend(finger_distances)
    
    # 中指の付け根(9)基準の指先の相対運動量（角度）: 5個
    finger_angles = [
        calculate_angle_3d(
            hand_landmarks.landmark[mid], 
            middle_root, 
            hand_landmarks.landmark[tip]
        ) 
        for mid, tip in zip(finger_mids, finger_tips)
    ]
    features.extend(finger_angles)
    
    # 手首(0)の運動量（距離，角度基準は9）: 1個
    wrist_middle_distance = calculate_distance(wrist, middle_root)
    features.append(wrist_middle_distance)
    
    # 距離全体の運動量: 1個
    total_distance = sum(finger_distances) / 5
    features.append(total_distance)
    
    # 角度全体の運動量: 1個
    total_angle = sum(finger_angles) / 5
    features.append(total_angle)
    
    # 指ごとのエネルギー: 5個
    finger_energies = [
        calculate_distance(
            hand_landmarks.landmark[tip], 
            hand_landmarks.landmark[root]
        )**2
        for tip, root in zip(finger_tips, finger_roots)
    ]
    features.extend(finger_energies)
    
    # 各指先のx方向、y方向、z方向の運動量（速度）: 5指 x 3方向 = 15個
    finger_momentums = []
    if prev_landmarks is not None:
        for tip in finger_tips:
            # 現在の位置
            current_pos = hand_landmarks.landmark[tip]
            # 前回の位置
            prev_pos = prev_landmarks.landmark[tip]
            
            # x, y, z方向の運動量（速度）を計算
            x_momentum = (current_pos.x - prev_pos.x) / frame_time
            y_momentum = (current_pos.y - prev_pos.y) / frame_time
            z_momentum = (current_pos.z - prev_pos.z) / frame_time
            
            finger_momentums.extend([x_momentum, y_momentum, z_momentum])
    else:
        # 初回フレームの場合は0を設定
        finger_momentums = [0] * (5 * 3)  # 5指 x 3方向
    
    features.extend(finger_momentums)

    # 左右移動量の計算（手首のx座標差分を利用）
    if prev_landmarks is not None:
        dx = hand_landmarks.landmark[0].x - prev_landmarks.landmark[0].x
        if dx > 0:
            right_movement = dx
            left_movement = 0
        else:
            right_movement = 0
            left_movement = -dx
    else:
        # 初回フレームは0
        right_movement = 0
        left_movement = 0

    left_right_ratio = right_movement / (left_movement + 1e-6)  # 0除算回避
    features.append(left_right_ratio)
    
    # 屈曲伸展比率（簡易的に親指先と根元の距離を屈曲指標と仮定）
    # 他の指も同様に応用可能
    thumb_tip = hand_landmarks.landmark[4]
    thumb_root = hand_landmarks.landmark[1]
    thumb_distance = calculate_distance(thumb_tip, thumb_root)
    
    # 前フレームの距離も取得して比較（伸展していれば距離増加）
    if prev_landmarks is not None:
        prev_thumb_tip = prev_landmarks.landmark[4]
        prev_thumb_root = prev_landmarks.landmark[1]
        prev_thumb_distance = calculate_distance(prev_thumb_tip, prev_thumb_root)
    else:
        prev_thumb_distance = thumb_distance

    # 屈曲している（閉じている）なら距離は小さく、伸展しているなら距離は大きいとみなすので、
    # 伸展時は距離の増加、屈曲時は減少とする簡易判定
    flexion_extension_ratio = (thumb_distance + 1e-6) / (prev_thumb_distance + 1e-6)
    features.append(flexion_extension_ratio)
    
    # === 相対ランドマーク ===
    # マーク0（手首）を基準とした相対座標で保存
    base_lm = hand_landmarks.landmark[0]
    for lm in hand_landmarks.landmark:
        rel_x = lm.x - base_lm.x
        rel_y = lm.y - base_lm.y
        rel_z = lm.z - base_lm.z
        features.extend([rel_x, rel_y, rel_z])
    
    # === Transformer_skeleton.py特徴量 ===
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
    # 距離
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
    # 角度
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
    # 重複しない特徴量のみ追加
    # adjacent_distances: 9個, 角度: 5+5+5+1+2=18個
    transformer_skeleton_features = []
    # adjacent_distancesのうち未収録分のみ追加
    transformer_skeleton_features.extend(adjacent_distances)
    # 角度類
    transformer_skeleton_features.extend([
        thumb_angle_from_palm, index_angle_from_palm, middle_angle_from_palm, ring_angle_from_palm, pinky_angle_from_palm,
        thumb_angle_at_vertex, index_angle_at_vertex, middle_angle_at_vertex, ring_angle_at_vertex, pinky_angle_at_vertex,
        thumb_angle_from_base, index_angle_from_base, middle_angle_from_base, ring_angle_from_base, pinky_angle_from_base,
        middleMCP_angle_from_base, thumb_xz_angle, pinky_xz_angle
    ])
    features.extend(transformer_skeleton_features)
    
    # 現在のランドマークを次回のために保存
    prev_landmarks = hand_landmarks
    
    return features

def process_video_and_annotations(video_path, annotation_path, output_base_dir):
    # 動画のファイル名（拡張子なし）
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 動画名のディレクトリ作成
    video_output_dir = os.path.join(output_base_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # アノテーションCSVの読み込み
    with open(annotation_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # ヘッダーをスキップ
        annotations = list(reader)
    
    # ビデオキャプチャ
    cap = cv2.VideoCapture(video_path)
    
    # 最終的な出力用のリスト
    all_transition_features = []
    all_segment_features = []
    
    # MediaPipe Handsの設定
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8) as hands:
        
        feature_headers = [
            'frame_number',
            'segment_id', 
            *[f'distance_tip_{i}' for i in range(5)],  # 手首基準の指先距離
            *[f'angle_tip_{i}' for i in range(5)],     # 中指根元基準の指先角度
            'wrist_middle_distance',                   # 手首-中指根元距離
            'total_distance',                          # 距離合計
            'total_angle',                             # 角度合計
            *[f'finger_energy_{i}' for i in range(5)],  # 指のエネルギー（コンマを追加）
            *[f'fingerx_energy_x{i}' for i in range(5)],
            *[f'fingerx_energy_y{i}' for i in range(5)],
            *[f'fingerx_energy_z{i}' for i in range(5)],
            'left_right_movement_ratio',                 # 左右移動量比率
            'flexion_extension_ratio',                    # 屈曲伸展比率
            *[f'landmark_{i}_x' for i in range(21)],
            *[f'landmark_{i}_y' for i in range(21)],
            *[f'landmark_{i}_z' for i in range(21)]
        ]
        
        # セグメント区間の処理
        for segment in annotations:
            segment_id, start_frame, end_frame, word, character = segment
            
            # セグメントのCSVファイルパス
            segment_output_path = os.path.join(
                video_output_dir, 
                f'segment_{segment_id}_{word}_features.csv'
            )
            
            # フレームの設定
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
            
            # セグメントの特徴量リスト
            segment_features = []
            
            # セグメント区間のフレーム処理
            for frame_num in range(int(start_frame), int(end_frame) + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # RGB変換
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 手の検出
                results = hands.process(frame_rgb)
                
                # フレームの特徴量
                frame_feature = [
                    frame_num, 
                    segment_id
                ]
                
                if results.multi_hand_landmarks:
                    # 最初の手の特徴量抽出
                    hand_landmarks = results.multi_hand_landmarks[0]
                    hand_features = extract_hand_features(hand_landmarks)
                    frame_feature.extend(hand_features)
                else:
                    # 手が検出されない場合は0で埋める
                    frame_feature.extend([0] * 35)
                
                segment_features.append(frame_feature)
            
            # セグメント区間の特徴量をCSVに保存
            if segment_features:
                with open(segment_output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(feature_headers)
                    writer.writerows(segment_features)
                
                print(f"セグメント区間特徴量保存: {segment_output_path}")
                
                # 全セグメント特徴量に追加
                all_segment_features.extend(segment_features)
        
        # 連続するアノテーションセグメント間の遷移区間も処理
        for i in range(len(annotations) - 1):
            current_segment = annotations[i]
            next_segment = annotations[i + 1]
            
            current_segment_id, current_start_frame, current_end_frame, _, _ = current_segment
            next_segment_id, next_start_frame, next_end_frame, _, _ = next_segment
            
            # 現在のセグメントの終了フレームから次のセグメントの開始フレームまでを遷移区間とする
            transition_start_frame = int(current_end_frame)
            transition_end_frame = int(next_start_frame)
            
            # 遷移区間のCSVファイルパス
            transition_output_path = os.path.join(
                video_output_dir, 
                f'transition_{current_segment_id}_to_{next_segment_id}_features.csv'
            )
            
            # フレームの設定
            cap.set(cv2.CAP_PROP_POS_FRAMES, transition_start_frame)
            
            # 遷移区間の特徴量リスト
            transition_features = []
            
            # 遷移区間のフレーム処理（既存のコードと同じ）
            for frame_num in range(transition_start_frame, transition_end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # RGB変換
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 手の検出
                results = hands.process(frame_rgb)
                
                # フレームの特徴量
                frame_feature = [
                    frame_num, 
                    current_segment_id, 
                    next_segment_id
                ]
                
                if results.multi_hand_landmarks:
                    # 最初の手の特徴量抽出
                    hand_landmarks = results.multi_hand_landmarks[0]
                    hand_features = extract_hand_features(hand_landmarks)
                    frame_feature.extend(hand_features)
                else:
                    # 手が検出されない場合は0で埋める
                    frame_feature.extend([0] * 17)
                
                transition_features.append(frame_feature)
            
            # 遷移区間の特徴量をCSVに保存
            if transition_features:
                with open(transition_output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # 遷移区間用のヘッダーを作成
                    transition_headers = [
                        'frame_number',
                        'from_segment', 
                        'to_segment',
                        *[f'distance_tip_{i}' for i in range(5)],
                        *[f'angle_tip_{i}' for i in range(5)],
                        'wrist_middle_distance',
                        'total_distance',
                        'total_angle',
                        *[f'finger_energy_{i}' for i in range(5)],
                        *[f'fingerx_energy_x{i}' for i in range(5)],
                        *[f'fingerx_energy_y{i}' for i in range(5)],
                        *[f'fingerx_energy_z{i}' for i in range(5)],
                        'left_right_movement_ratio',   # 追加
                        'flexion_extension_ratio'      # 追加
                    ]
                    writer.writerow(transition_headers)
                    writer.writerows(transition_features)
                
                print(f"遷移区間特徴量保存: {transition_output_path}")
                
                # 全遷移特徴量に追加
                all_transition_features.extend(transition_features)
    
    # 全セグメントの特徴量を保存（オプション）
    if all_segment_features:
        combined_segment_path = os.path.join(
            output_base_dir, 
            f'{video_name}_all_segments_features.csv'
        )
        with open(combined_segment_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(feature_headers)
            writer.writerows(all_segment_features)
        
        print(f"全セグメント特徴量保存: {combined_segment_path}")
    
    # 全遷移の特徴量を保存（オプション）
    if all_transition_features:
        combined_transition_path = os.path.join(
            output_base_dir, 
            f'{video_name}_all_transitions_features.csv'
        )
        with open(combined_transition_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(transition_headers)
            writer.writerows(all_transition_features)
        
        print(f"全遷移特徴量保存: {combined_transition_path}")

def process_all_videos():
    # 対象ディレクトリリスト
    input_dirs = [
        '../Video/transformer訓練/動作指文字',
        '../Video/transformer訓練/静止指文字',
    ]
    output_base_dir = '../訓練データ/単語分割'
    os.makedirs(output_base_dir, exist_ok=True)
    for input_dir in input_dirs:
        # 動作指文字 or 静止指文字のディレクトリ名を取得
        if '動作指文字' in input_dir:
            sub_dir = '動作指文字'
        elif '静止指文字' in input_dir:
            sub_dir = '静止指文字'
        else:
            sub_dir = 'その他'
        output_dir = os.path.join(output_base_dir, sub_dir)
        os.makedirs(output_dir, exist_ok=True)
        for folder_name in sorted(os.listdir(input_dir)):
            folder_path = os.path.join(input_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            for filename in sorted(os.listdir(folder_path)):
                if filename.endswith(('.mp4', '.mov')):
                    video_path = os.path.join(folder_path, filename)
                    video_name = os.path.splitext(filename)[0]
                    # 出力csv名を「動画名.csv」に変更
                    output_csv = os.path.join(output_dir, f'{video_name}.csv')
                    cap = cv2.VideoCapture(video_path)
                    frame_idx = 0
                    features_list = []
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
                            row = [frame_idx]
                            if results.multi_hand_landmarks:
                                hand_landmarks = results.multi_hand_landmarks[0]
                                feats = extract_hand_features(hand_landmarks)
                                row.extend(feats)
                            else:
                                row.extend([0]* (5+5+1+1+1+5+15+1+1+21*3+9+18))
                            features_list.append(row)
                            frame_idx += 1
                    cap.release()
                    # ヘッダー作成
                    feature_headers = ['frame_number']
                    feature_headers += [f'distance_tip_{i}' for i in range(5)]
                    feature_headers += [f'angle_tip_{i}' for i in range(5)]
                    feature_headers += ['wrist_middle_distance','total_distance','total_angle']
                    feature_headers += [f'finger_energy_{i}' for i in range(5)]
                    feature_headers += [f'fingerx_energy_x{i}' for i in range(5)]
                    feature_headers += [f'fingerx_energy_y{i}' for i in range(5)]
                    feature_headers += [f'fingerx_energy_z{i}' for i in range(5)]
                    feature_headers += ['left_right_movement_ratio','flexion_extension_ratio']
                    # === ここから相対ランドマーク ===
                    feature_headers += [f'rel_landmark_{i}_x' for i in range(21)]
                    feature_headers += [f'rel_landmark_{i}_y' for i in range(21)]
                    feature_headers += [f'rel_landmark_{i}_z' for i in range(21)]
                    # === ここからTransformer_skeleton.py特徴量 ===
                    feature_headers += [f'adj_dist_{i}' for i in range(9)]
                    feature_headers += [f'angle_{i}' for i in range(18)]
                    with open(output_csv, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(feature_headers)
                        writer.writerows(features_list)
                    print(f'保存: {output_csv}')

if __name__ == '__main__':
    process_all_videos()