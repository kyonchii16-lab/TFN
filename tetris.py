import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from e3nn import o3, nn as e3nn_nn
from e3nn import math as e3nn_math
from typing import List, Tuple
from torch.optim.lr_scheduler import CosineAnnealingLR

# ==========================================
# 1. データセット生成 (3D Tetris Shapes)
# ==========================================
class TetrisDataset:
    """
    3Dテトリス（テトロキューブ）の8形状を生成するクラス。
    各形状は4つのブロック（点）からなる。
    """
    def __init__(self):
        # 8種類のテトロキューブの座標 (中心化済みと仮定)
        # 1. I (Straight)
        # 2. O (Square)
        # 3. L
        # 4. J (Mirror of L)
        # 5. T
        # 6. S
        # 7. Z (Mirror of S)
        # 8. Corner (Tripod)

        self.shapes = [
            [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
            [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)], # chiral_shape_2
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
            [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # T
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # zigzag
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]   # L
            ]

        # テンソル化と中心化（重心を原点に）
        self.shape_tensors = []
        for s in self.shapes:
            t = torch.tensor(s, dtype=torch.float32)
            t = t - t.mean(dim=0, keepdim=True) # 中心化
            self.shape_tensors.append(t)

    def get_batch(self, batch_size=32, random_rotate=False):
        """
        バッチデータを生成する。
        各サンプルについてランダムな形状を選び、必要なら回転させる。
        """
        batch_pos = []
        batch_feat = []  # 入力特徴量 (全て1のスカラ)
        batch_indices = [] # どのバッチに属するか
        labels = []

        total_points = 0

        for b in range(batch_size):
            idx = b % 8
            pos = self.shape_tensors[idx].clone()

            # ランダム回転の適用 (SO(3))
            if random_rotate:
                # e3nnのランダム回転行列
                rot = o3.rand_matrix()
                pos = pos @ rot.T

            batch_pos.append(pos)
            # 特徴量はスカラー1 (mass/occupancy)
            batch_feat.append(torch.ones((4, 1), dtype=torch.float32))
            batch_indices.append(torch.full((4,), b, dtype=torch.long))
            labels.append(idx)
            total_points += 4

        return {
            'pos': torch.cat(batch_pos, dim=0),      # [B*4, 3]
            'x': torch.cat(batch_feat, dim=0),       # [B*4, 1]
            'batch': torch.cat(batch_indices, dim=0),# [B*4]
            'y': torch.tensor(labels, dtype=torch.long) # [B]
        }

# ==========================================
# 2. TFN Layer Implementation (e3nn)
# ==========================================
class TFNLayer(torch.nn.Module):
    """
    Tensor Field Networksの1レイヤー (Figure 3のモジュールに相当)。
    Convolution -> Self-Interaction -> Nonlinearity
    """
    def __init__(self, irreps_in, irreps_sh, instructions, irreps_conv, irreps_out, max_radius=3.5, num_basis=4):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.instructions = instructions
        self.irreps_conv = o3.Irreps(irreps_conv)
        self.irreps_out = o3.Irreps(irreps_out)
        self.max_radius = max_radius
        self.num_basis = num_basis
        #self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)
        # --- 1. Convolution (Tensor Product with Radial Weights) ---
        self.tp = o3.TensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_conv,
            # 自分で計算パス（instruction）を定義する
            instructions=self.instructions,
            shared_weights=False,
            internal_weights=False
        )

        # 動径関数 (Radial Function): 距離 -> TensorProductの重み
        # MLP: Distance -> Weights for TP
        self.fc_rad = e3nn_nn.FullyConnectedNet(
            [1, 16, self.tp.weight_numel],
            act=torch.nn.functional.silu
        )

        # self.fc_rad = torch.nn.Sequential(
        #     torch.nn.Linear(num_basis, 16), # soft_one_hotされた後の次元数からスタート
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(16, self.tp.weight_numel)
        # )

        # --- 2. Self-Interaction (Channel Mixing) ---
        # 各Irrep内でのチャンネル混合 (Linear)
        # o3.Linear: 同じirrepsを持つチャンネル同士だけを混ぜる
        self.linear = o3.Linear(self.irreps_conv, self.irreps_out)

        # --- 3. Nonlinearity ---
        # 論文では Norm-based nonlinearity を使用: eta(||x||) * x
        # e3nnでは NormActivation がこれに相当
        # l=0にはスカラー活性化、l>0にはノルムに対する活性化を適用
        self.nonlinearity = e3nn_nn.NormActivation(
            self.irreps_out,
            torch.nn.functional.silu, # scalar activation
            torch.sigmoid             # gate/norm activation
        )

    def forward(self, x, pos, batch):
        """
        x: [N, Cin] node features
        pos: [N, 3] node positions
        batch: [N] batch indices
        """
        # エッジの作成 (今回は各形状内での全結合グラフとする)
        # 同じバッチ内のノード間のみエッジを張る
        edge_index = self._build_fully_connected_edges(batch)
        src, dst = edge_index

        # 相対ベクトル
        vec = pos[src] - pos[dst]
        dist = vec.norm(dim=1, keepdim=True)

        # 球面調和関数 Y(vec)
        edge_sh = o3.spherical_harmonics(
            self.irreps_sh,
            vec,
            normalize=True, # 入力（相対ベクトル）を正規化
            normalization='component'
        )

        # 距離をソフトOne-hot展開する
        # dist: [E, 1] -> radial_basis: [E, num_basis]
        # radial_basis = e3nn_math.soft_one_hot_linspace(
        #     dist,
        #     start=0.0,
        #     end=self.max_radius,
        #     number=self.num_basis,
        #     basis='gaussian',
        #     cutoff=False
        # )

        # 動径関数の重み
        # edge_weights = self.fc_rad(radial_basis) # fc_rad = Fully Connected Radial

        edge_weights = self.fc_rad(dist)

        # Convolution (Tensor Product)
        # input[src] (x) Y[edge] * weights[edge]
        out_edge = self.tp(x[src], edge_sh, edge_weights)

        # Aggregate (Sum pooling over neighbors)
        out_node = torch.zeros_like(x, dtype=out_edge.dtype)
        # 出力の次元が入力と異なる可能性があるためサイズ調整
        if out_node.shape[1] != out_edge.shape[1]:
            out_node = torch.zeros((x.shape[0], out_edge.shape[1]), device=x.device)

        out_node.index_add_(0, dst, out_edge)

        # Self-Interaction
        out_node = self.linear(out_node)

        # Nonlinearity
        out_node = self.nonlinearity(out_node)

        return out_node

    def _build_fully_connected_edges(self, batch):
        # バッチ内の全結合エッジを作成する簡易関数
        dev = batch.device
        N = batch.shape[0]
        # ブロードキャストでマスク作成
        mask = batch.view(-1, 1) == batch.view(1, -1)
        # 自分自身へのループは除外する場合が多いが、TFN論文では含めることもある
        # ここでは除外しておく
        mask.fill_diagonal_(False)

        src, dst = torch.nonzero(mask, as_tuple=True)
        return torch.stack([src, dst])

# ==========================================
# 3. Tetris TFN Model (Figure 3 Architecture)
# ==========================================
class TetrisTFN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # --- Figure 3の設定 ---
        # Input: Scalar (1 channel, l=0) -> '1x0e'
        # Layer 1: Outputs 4 channels of l=0 and 4 channels of l=1
        # Layer 2: Outputs 4 channels of l=0 and 4 channels of l=1
        # Layer 3: Outputs 4 channels of l=0 (and l=1 discarded or kept)
        # Global Pool: Sum
        # Softmax: Output 8 classes

        # Layer 1
        self.irreps_1_in   = o3.Irreps("1x0e")
        self.irreps_1_sh   = o3.Irreps("1x0e + 1x1e")
        self.instructions_1 = [
                (0, 0, 0, 'uuu', True), # (in1_idx, in2_idx, out_idx, mode, has_weight)
                (0, 1, 1, 'uuu', True)
            ]
        self.irreps_1_conv = o3.Irreps("1x0e + 1x1e")
        self.irreps_1_out  = o3.Irreps("4x0e + 4x1e")

        # Layer 2
        self.irreps_2_in   = self.irreps_1_out
        self.irreps_2_sh   = o3.Irreps("4x0e + 4x1e")
        self.instructions_2 = [
                (0, 0, 0, 'uuu', True), # (in1_idx, in2_idx, out_idx, mode, has_weight)
                (1, 1, 1, 'uuu', True),
                (1, 0, 2, 'uuu', True),
                (0, 1, 3, 'uuu', True),
                (1, 1, 4, 'uuu', True)
            ]
        self.irreps_2_conv = o3.Irreps('4x0e + 4x0e + 4x1e + 4x1e + 4x1e')
        self.irreps_2_out  = o3.Irreps("4x0e + 4x1e")

        # Layer 3
        self.irreps_3_in   = self.irreps_2_out
        self.irreps_3_sh   = o3.Irreps("4x0e + 4x1e")
        self.instructions_3 = [
                (0, 0, 0, 'uuu', True), # (in1_idx, in2_idx, out_idx, mode, has_weight)
                (1, 1, 1, 'uuu', True)
            ]
        self.irreps_3_conv = o3.Irreps("4x0e + 4x0e")
        self.irreps_3_out  = o3.Irreps("4x0e")

        self.irreps_hidden = self.irreps_3_out

        # --- モデル構築 ---

        # Layer 1
        self.layer1 = TFNLayer(self.irreps_1_in, self.irreps_1_sh, self.instructions_1, self.irreps_1_conv, self.irreps_1_out)

        # Layer 2
        self.layer2 = TFNLayer(self.irreps_2_in, self.irreps_2_sh, self.instructions_2, self.irreps_2_conv, self.irreps_2_out)

        # Layer 3
        self.layer3 = TFNLayer(self.irreps_3_in, self.irreps_3_sh, self.instructions_3, self.irreps_3_conv, self.irreps_3_out)

        # Output Projection
        # Global Pool後に l=0 の成分(4次元)を 8クラスに射影
        # self.irreps_hidden['0e'] は最初の4次元
        self.final_linear = nn.Linear(4, 8)

    def forward(self, pos, x, batch):
        """
        pos: [N, 3]
        x: [N, 1]
        batch: [N]
        """
        # --- Layer 1 ---
        h = self.layer1(x, pos, batch)

        # --- Layer 2 ---
        h = self.layer2(h, pos, batch)

        # --- Layer 3 ---
        h = self.layer3(h, pos, batch)

        # --- Extract l=0 features ---
        # irreps "4x0e + 4x1o" のうち、最初の "4x0e" だけを取り出す
        idx_scalar = self.irreps_hidden.slices()[0] # slice object for 4x0e
        h_scalar = h[:, idx_scalar] # [N, 4]

        # --- Global Pooling (Sum) ---
        # バッチごとに和を取る
        num_graphs = batch.max().item() + 1
        out = torch.zeros((num_graphs, h_scalar.shape[1]), device=h.device)
        out.index_add_(0, batch, h_scalar)

        # --- Softmax (Logits) ---
        logits = self.final_linear(out)

        return logits

# ==========================================
# 4. 学習・評価ループ
# ==========================================
def main():
    print("Initializing TFN for Tetris Shape Classification...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # データセットとモデル
    dataset = TetrisDataset()
    model = TetrisTFN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # 学習の最後(1500ステップ)に向けて、学習率を徐々に 0 に近づける設定
    #scheduler = CosineAnnealingLR(optimizer, T_max=1501)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting training (Training on FIXED shapes)...")

    steps = 2000
    batch_size = 8

    for step in range(steps):
        model.train()

        # バッチ取得 (ランダム回転なし)
        batch_data = dataset.get_batch(batch_size, random_rotate=False)
        pos = batch_data['pos'].to(device)
        x = batch_data['x'].to(device)
        batch = batch_data['batch'].to(device)
        y = batch_data['y'].to(device)

        optimizer.zero_grad()
        logits = model(pos, x, batch)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            # 精度確認
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean().item()
            print(f"Step {step:03d} | Loss: {loss.item():.4f} | Acc: {acc*100:.1f}%")

    print("\nTraining Finished.")
    print("Testing on NEW Randomly Rotated Shapes...")

    model.eval()
    test_batch = dataset.get_batch(100, random_rotate=True)
    pos = test_batch['pos'].to(device)
    x = test_batch['x'].to(device)
    batch = test_batch['batch'].to(device)
    y = test_batch['y'].to(device)

    with torch.no_grad():
        logits = model(pos, x, batch)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean().item()

    print(f"Test Accuracy (Random Rotations): {acc*100:.1f}%")

    if acc > 0.9:
        print("Success! The TFN correctly classifies rotated 3D Tetris shapes.")
    else:
        print("Accuracy is low. Check hyperparameters or training duration.")

if __name__ == "__main__":
    main()
