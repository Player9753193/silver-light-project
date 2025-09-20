import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
import os
from student_model import MLPWithAttention

# KL 散度蒸馏损失
class KDLoss(nn.Module):
    def __init__(self, temperature=4.0):
        super(KDLoss, self).__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_probs):
        student_log_prob = nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        teacher_prob = teacher_probs / teacher_probs.sum(dim=1, keepdim=True)  # 归一化
        teacher_prob = teacher_prob / self.temperature
        teacher_prob = nn.functional.softmax(teacher_prob, dim=1)
        loss = self.kl_div(student_log_prob, teacher_prob) * (self.temperature ** 2)
        return loss


def load_data(data_dir, device, val_split=0.2):
    X = np.load(f"{data_dir}/X_features.npy")  # (N, 11)
    Y_soft = np.load(f"{data_dir}/Y_soft_labels.npy")  # (N, 8)

    # 归一化软标签
    Y_soft = Y_soft / 2.0
    Y_soft = torch.softmax(torch.tensor(Y_soft, dtype=torch.float32), dim=1).numpy()

    # 转为 tensor
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_soft_tensor = torch.tensor(Y_soft, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_tensor, Y_soft_tensor)

    # 划分训练集和验证集
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader


def plot_loss_curve(train_losses, val_losses, save_path="loss_curve.png"):
    """绘制并保存训练/验证损失曲线"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r--', label='Validation Loss', linewidth=2)

    plt.title('Knowledge Distillation Training Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('KL Divergence Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # 可视化图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 损失曲线已保存: {save_path}")
    plt.show()


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./student_data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_model", type=str, default="student_mlp_attn.pth")
    parser.add_argument("--patience", type=int, default=15, help="早停耐心值")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    # 数据加载（含验证集）
    train_loader, val_loader = load_data(args.data_dir, device)

    # 模型
    model = MLPWithAttention(input_dim=11, hidden_dim=64, num_classes=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = KDLoss(temperature=4.0)

    # 早停机制
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    # 训练循环
    model.train()
    for epoch in range(args.epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        for x, y_soft in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y_soft)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y_soft in val_loader:
                logits = model(x)
                loss = criterion(logits, y_soft)
                val_loss += loss.item()
            val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # --- 早停判断 ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), args.save_model)
            print(f"✅ 模型保存: {args.save_model} (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        # --- 打印进度 ---
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{args.epochs}] | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Patience: {patience_counter}/{args.patience}")

        # 早停
        if patience_counter >= args.patience:
            print(f"📢 早停触发！最佳验证损失: {best_val_loss:.4f}")
            break

    # --- 绘制损失曲线 ---
    plot_loss_curve(train_losses, val_losses)

    print(f"🎉 训练完成！最终模型保存为: {args.save_model}")


if __name__ == "__main__":
    train()