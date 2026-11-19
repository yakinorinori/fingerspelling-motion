# fingerspelling-motion — 動作モデル（手の動き 5クラス分類）

1D-CNN と Transformer を、特徴量 2 種類（既存125次元・＋案A）で比較する。

## フォルダの置き場所

```
python_画像処理/
├── Fingerspelling/Video/transformer訓練/手の動き 5クラス分類/クラス0〜4/かな/動画
└── Fingerspelling-repos/
    └── fingerspelling-motion/  ← このリポジトリ（git clone した場所）
        ├── extract_landmarks.py   手順1：動画 → ランドマーク
        ├── train_compare.py       手順2：4つの組み合わせを交差検証で比較
        ├── recognize_video.py     手順3：実際の動画で認識結果を見る
        ├── features.py            特徴量（既存125次元・案A 18次元）
        ├── models.py              1D-CNN・Transformer
        ├── common.py              パスなどの共通設定
        └── 元のコード/            Fingerspelling-word-division からのコピー（参照用）
```

## 使い方

```bash
cd Fingerspelling-repos/fingerspelling-motion

# 手順1：MediaPipe でランドマークを取り出して保存（最初の1回だけ時間がかかる）
python extract_landmarks.py

# 手順2：比較（5分割交差検証）＋ 全データで学習したモデルの保存
python train_compare.py --final

# 手順3：実際の動画で確認
python recognize_video.py --model results/models/cnn_base+A.pt 動画.mp4
```

- データの場所が違うときは `extract_landmarks.py --data "（フォルダのパス）"`
- 人ごとに分けて評価したいとき（同じ人が学習とテストの両方に入らないように）:
  `train_compare.py --group-regex "^(ASL_[A-E]|ubmoji)"` のように、ファイル名から人などを取り出す正規表現を指定する
- 結果は `results/`（summary.csv、混同行列の画像、モデル、ラベル付き動画）

## 比較の条件

| | 1D-CNN | Transformer |
|---|---|---|
| 構造 | dilated 1D-CNN（k=3, dilation 1,2,4,8, 受容野31フレーム, 64ch, 残差接続） | 入力を64次元に線形変換 → 位置エンコーディング → Encoder 2層（4ヘッド） |
| 出力 | 本物のフレームだけの平均 → 全結合 → 5クラス | 同左 |

共通：標準化（学習データのみで計算）、Adam（lr 1e-3）、クラスの本数の偏りを損失の重みで補正、
学習データの15%を検証用に取り分けて最良エポックを選ぶ、全組み合わせで同じ分割を使う。

元の TransformerModel.py との違い：特徴量の次元数が nhead で割り切れなくてもよいように入力を線形変換、
パディング部分をマスク、平均にパディングを含めない。

## 特徴量

### base（既存の125次元）
`元のコード/eval_transformer.py` の `extract_hand_features` と同じ計算。直した点は2つ。
- 前フレームの記憶を動画ごとにリセット（元はグローバル変数で、前の動画の最後のフレームが残っていた）
- 手が無いフレームは125個の0で埋める（元は35個や17個で、行の長さがそろっていなかった）

### 案A（動きの方向、18次元）
手のひら中心（0,5,9,13,17番の平均）の動きを、手の大きさ（手首→中指の付け根）で割って使う。
座標は画素単位に直す（正規化座標のままだと横長の動画で横の動きが縮むため）。
前後7フレーム（約0.5秒）の窓でまとめる。y は上を正にしている。

| 名前 | 内容 | 横 | 縦 | 斜め |
|---|---|---|---|---|
| dir8_E〜SE（8） | 移動方向の8方向ヒストグラム（移動量で重み付け） | E/W | N/S | NE/NW/SE/SW |
| hv_ratio | Σdx² ÷ (Σdx²+Σdy²) | 1 | 0 | 0.5付近 |
| consistency_x, _y | Σdx ÷ Σ\|dx\|（往復なら0） | x=±1 | y=±1 | 両方±1 |
| straightness | 始点→終点 ÷ 道のり | 直線なら1 | | |
| pca_cos2, pca_sin2, linearity | 軌跡の主方向（2倍角で表す）と直線らしさ | | | |
| rotation | 手首→中指の付け根の向きの変化（手首のひねり） | | | |
| scale_change | 手の大きさの変化（手前・奥への動き） | | | |
| path_length | 窓内の移動量 | | | |

## 案B（今回は作らない。アイデアとして残す）

画像処理で動きの方向を数値にする方法。計算が重いので今回は見送り。
骨格が取れないフレーム（速い動きでぶれたとき）を補う役割で、将来比較する価値がある。

- **B1. オプティカルフローの方向別の面積**：手の周りの矩形で Farneback 法などにより画素ごとの動き (u,v) を求め、
  「|u| が |v| より十分大きく、一定以上動いた画素」の面積を水平成分として合計（垂直・斜めも同様）、矩形の面積で割る。
  行動認識の HOF（Histogram of Optical Flow）にあたる。
- **B2. 動きの履歴画像（MHI）**：骨格だけを描いた黒背景の動画（`VideoBlack.py`）で、動いた画素ほど明るく残る画像を作り、
  モーメントから向きと縦横比を出す。横に動けば横長、縦に動けば縦長の跡になる。
- **B3. 軌跡画像の形**：手首の軌跡を小さな画像に描き、画像モーメントで向き・細長さを出す。の・り・ん のようななぞる動きに強い。
