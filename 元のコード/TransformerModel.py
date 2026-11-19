import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
import math
import os
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# データセットクラス
class SignLanguageDataset:
    def __init__(self, feature_dir, target_file):
        # ターゲットファイルの読み込み
        if not os.path.exists(target_file):
            raise FileNotFoundError(f'ターゲットファイルが見つかりません: {target_file}')
        try:
            all_labels = pd.read_csv(target_file, header=None)[0].values
        except pd.errors.EmptyDataError:
            raise ValueError(f'ターゲットファイルが空です: {target_file}')
        self.data = []
        self.labels = []
        idx = 0
        for root, _, files in os.walk(feature_dir):
            for filename in sorted(files):
                if filename.endswith('.csv'):
                    file_path = os.path.join(root, filename)
                    if not os.path.exists(file_path):
                        print(f'ファイルが見つかりません: {file_path}')
                        continue
                    try:
                        df = pd.read_csv(file_path)
                        if df.empty:
                            print(f'ファイルが空です: {file_path}')
                            continue
                        # 1列目（フレームナンバー）を除外
                        self.data.append(torch.tensor(df.iloc[:, 1:].values, dtype=torch.float))
                        if idx < len(all_labels):
                            self.labels.append(all_labels[idx])
                        else:
                            print(f'ラベル数が足りません: {target_file} {filename}')
                        idx += 1
                    except Exception as e:
                        print(f'ファイル読み込み中にエラーが発生しました: {file_path}, エラー: {e}')
        print(f"{feature_dir}: 読み込んだcsv数={len(self.data)}, ラベル数={len(self.labels)}")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if idx < len(self.data):
            return self.data[idx], self.labels[idx]
        else:
            raise IndexError("Index out of range for dataset")

# データのバッチ処理用関数
def collate_fn(batch):
    data, labels = zip(*batch)
    data_padded = pad_sequence(data, batch_first=True, padding_value=0.0)  # パディング
    return data_padded, torch.tensor(labels, dtype=torch.long)

# 位置エンコーディング
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Transformerモデル
class TimeSeriesTransformer(nn.Module):
    def __init__(self, feature_size=98, num_classes=41, seq_length=100, num_layers=2, nhead=9, dim_feedforward=512):
        super(TimeSeriesTransformer, self).__init__()
        self.pos_encoder = PositionalEncoding(feature_size, 0.1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = src.transpose(0, 1)  # (batch_size, seq_length, feature_size) -> (seq_length, batch_size, feature_size)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=0)  # (seq_length, batch_size, feature_size) -> (batch_size, feature_size)
        output = self.decoder(output)
        return output

# メイン関数
def main():
    # 3クラス分類用に3つのディレクトリ・ターゲットファイルをまとめて扱う
    feature_dirs = [
        '../訓練データ/単語分割/動作指文字',
        '../訓練データ/単語分割/静止指文字',
        '../訓練データ/単語分割/文字間'
    ]
    target_files = [
        '../訓練データ/単語分割/target_dynamic.csv',
        '../訓練データ/単語分割/target_static.csv',
        '../訓練データ/単語分割/target_between.csv'
    ]
    num_classes = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # すべての特徴量・ラベルを結合
    all_features = []
    all_labels = []
    for feature_dir, target_file in zip(feature_dirs, target_files):
        dataset = SignLanguageDataset(feature_dir, target_file)
        for i in range(len(dataset)):
            feats, label = dataset[i]
            all_features.append(feats)
            all_labels.append(label)
    # データセット作成
    class AllDataset(torch.utils.data.Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]
    full_dataset = AllDataset(all_features, all_labels)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # モデルの定義
    # feature_size=126 なら nhead=2,3,6,7,9,14,18,21,42,63,126 などで割り切れる値のみOK
    model = TimeSeriesTransformer(
        feature_size=125,        # 必要に応じて修正
        num_classes=num_classes,
        seq_length=200,
        num_layers=2,            # 通常2~4程度で十分
        nhead=5,                 # 125÷5=25で割り切れる
        dim_feedforward=512
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    # ロスと精度を保存するリスト
    train_losses = []
    val_losses = []
    val_accuracies = []

    # 学習ループ
    best_val_acc = -1
    best_epoch = -1
    best_model_state = None
    for epoch in range(100):  # エポック数を100に変更
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f'Epoch {epoch+1}, Loss: {avg_train_loss:.4f}')
        # 検証
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / len(val_dataset)
        val_losses.append(avg_val_loss)
        val_accuracies.append(accuracy)
        print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')
        # 最良モデルの保存
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
    # 最良モデルのみ保存
    if best_model_state is not None:
        best_model_path = f'./best_model.pth'
        torch.save(best_model_state, best_model_path)
        print(f'Best model (epoch {best_epoch}, acc={best_val_acc:.4f}) saved to {best_model_path}')
    # グラフの表示
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()