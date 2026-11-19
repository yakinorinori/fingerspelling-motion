"""動作モデル共通：パス、フォルダ探索、データ一覧の読み込み"""
import csv
import re
import unicodedata
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# Fingerspelling-repos/動作モデル/ から見た Fingerspelling/Video/transformer訓練/手の動き 5クラス分類
DEFAULT_DATA_DIR = (SCRIPT_DIR.parent.parent / "Fingerspelling" / "Video"
                    / "transformer訓練" / "手の動き 5クラス分類")
CACHE_DIR = SCRIPT_DIR / "cache"
LANDMARK_DIR = CACHE_DIR / "landmarks"
INDEX_CSV = CACHE_DIR / "index.csv"
RESULT_DIR = SCRIPT_DIR / "results"

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".m4v", ".mkv"}
NUM_CLASSES = 5
# 動画に表示するクラス名（cv2 は日本語を描けないので英数字で）
CLASS_NAMES = {i: f"Class {i}" for i in range(NUM_CLASSES)}


def norm(name):
    """macOS の濁点の分解（NFD）と全角・半角スペースの違いを吸収する"""
    return unicodedata.normalize("NFC", name).replace("　", "").replace(" ", "")


def resolve_dir(path):
    """表記ゆれを吸収して実在するフォルダを探す（見つからなければ None）"""
    path = Path(path).expanduser()
    if path.is_dir():
        return path.resolve()
    cur = Path(path.anchor) if path.is_absolute() else Path(".")
    for part in path.parts[1:] if path.is_absolute() else path.parts:
        if (cur / part).is_dir():
            cur = cur / part
            continue
        match = [d for d in cur.iterdir() if d.is_dir() and norm(d.name) == norm(part)] if cur.is_dir() else []
        if not match:
            return None
        cur = match[0]
    return cur.resolve()


def class_label(folder_name):
    """'クラス3' や 'class3' から 3 を取り出す"""
    m = re.search(r"\d+", unicodedata.normalize("NFKC", folder_name))
    return int(m.group()) if m else None


def load_index():
    """extract_landmarks.py が作った一覧（1行 = 1動画）"""
    if not INDEX_CSV.exists():
        raise SystemExit(f"{INDEX_CSV} がありません。先に extract_landmarks.py を実行してください")
    with open(INDEX_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
    return rows
