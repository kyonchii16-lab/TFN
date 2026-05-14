# Tensor Field Network (TFN) の実装

`e3nn`ライブラリを用いてゼロから構築した Tensor Field Network (Thomas et al., 2018) のPyTorch実装です。
3Dテトリスブロックの形状分類タスクにおいて、完全な**SE(3)同変性**を実現しています。
TFN論文におけるネットワーク構造をe3nnを用いて忠実に再現しました。テンソル積の計算パス（instructions='uuu'）を明示的に定義し、空間方向の畳み込み（球面調和関数）とチャンネル方向の混合（o3.Linear）を分離する設計思想を確実に再現しました。

- **SE(3)同変性の証明**
  固定された向きの形状のみで学習を行い、**データ拡張を一切使用せずに**、未知のランダムな回転に対するテストで100%の精度を達成しました。

## 実行方法

```bash
pip install -r requirements.txt
python tetris.py
```

## 実行例

```bash
Initializing TFN for Tetris Shape Classification...
Starting training (Training on FIXED shapes)...
Step 000 | Loss: 41.6521 | Acc: 12.5%
Step 050 | Loss: 3.3470 | Acc: 12.5%
Step 100 | Loss: 2.3376 | Acc: 12.5%
Step 150 | Loss: 2.2089 | Acc: 12.5%
Step 200 | Loss: 2.1413 | Acc: 12.5%
Step 250 | Loss: 2.0793 | Acc: 12.5%
Step 300 | Loss: 2.0067 | Acc: 12.5%
Step 350 | Loss: 1.9306 | Acc: 25.0%
Step 400 | Loss: 1.8098 | Acc: 37.5%
Step 450 | Loss: 1.6333 | Acc: 37.5%
Step 500 | Loss: 1.4444 | Acc: 50.0%
Step 550 | Loss: 1.2427 | Acc: 50.0%
Step 600 | Loss: 1.0340 | Acc: 62.5%
Step 650 | Loss: 0.8251 | Acc: 75.0%
Step 700 | Loss: 0.6600 | Acc: 75.0%
Step 750 | Loss: 0.5551 | Acc: 87.5%
Step 800 | Loss: 0.4883 | Acc: 75.0%
Step 850 | Loss: 0.4344 | Acc: 87.5%
Step 900 | Loss: 0.3811 | Acc: 87.5%
Step 950 | Loss: 0.3290 | Acc: 87.5%
Step 1000 | Loss: 0.2848 | Acc: 87.5%
Step 1050 | Loss: 0.2526 | Acc: 87.5%
Step 1100 | Loss: 0.2309 | Acc: 87.5%
Step 1150 | Loss: 0.2164 | Acc: 87.5%
Step 1200 | Loss: 0.2063 | Acc: 100.0%
Step 1250 | Loss: 0.1986 | Acc: 100.0%
Step 1300 | Loss: 0.1922 | Acc: 100.0%
Step 1350 | Loss: 0.1866 | Acc: 100.0%
Step 1400 | Loss: 0.1819 | Acc: 100.0%
Step 1450 | Loss: 0.1778 | Acc: 100.0%
Step 1500 | Loss: 0.1741 | Acc: 100.0%
Step 1550 | Loss: 0.1706 | Acc: 100.0%
Step 1600 | Loss: 0.1674 | Acc: 100.0%
Step 1650 | Loss: 0.1643 | Acc: 100.0%
Step 1700 | Loss: 0.1613 | Acc: 100.0%
Step 1750 | Loss: 0.1583 | Acc: 100.0%
Step 1800 | Loss: 0.1555 | Acc: 100.0%
Step 1850 | Loss: 0.1527 | Acc: 100.0%
Step 1900 | Loss: 0.1499 | Acc: 100.0%
Step 1950 | Loss: 0.1472 | Acc: 100.0%

Training Finished.
Testing on NEW Randomly Rotated Shapes...
Test Accuracy (Random Rotations): 100.0%
Success! The TFN correctly classifies rotated 3D Tetris shapes.
```
