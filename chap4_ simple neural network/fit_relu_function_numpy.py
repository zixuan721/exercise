
"""fit_relu_function_numpy.py

Two-layer ReLU network implemented with NumPy and manual backprop.

This script trains a 1-D regression model to fit a user-defined target function.
It is meant to satisfy the "函数拟合"要求（两层ReLU网络）并且不使用任何深度学习框架。

运行示例:
    python fit_relu_function_numpy.py

输出:
    - 训练/测试损失打印
    - 会在代码所在同级目录生成 `fit_relu_function.png`，展示拟合曲线
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 函数定义与数据采集
# ==========================================

# 定义目标函数 f(x) = sin(pi * x) + 0.5 * x^2
def target_function(x):
    return np.sin(np.pi * x) + 0.5 * (x ** 2)

# 设置随机种子，保证结果可复现
np.random.seed(42)

# 生成训练集 (带噪声)
n_train = 2000
# 在 [-3, 3] 区间内随机均匀采样
X_train = np.random.uniform(-3, 3, (n_train, 1))
# 计算真实 y 值并加入正态分布噪声
noise = np.random.normal(0, 0.1, (n_train, 1))
Y_train = target_function(X_train) + noise

# 生成测试集 (无噪声，用于评估真实拟合效果)
n_test = 500
X_test = np.linspace(-3, 3, n_test).reshape(-1, 1)
Y_test = target_function(X_test)

# ==========================================
# 2. 模型定义与初始化 (两层 ReLU 网络)
# ==========================================

input_dim = 1      # 输入层维度 (x)
hidden_dim = 256   # 隐藏层神经元个数
output_dim = 1     # 输出层维度 (y)

# 权重初始化：使用 He Initialization，适合 ReLU 激活函数
W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
b2 = np.zeros((1, output_dim))

# ==========================================
# 3. Adam 优化器参数初始化
# ==========================================
# 手动实现 Adam 优化器以加快收敛速度并提高拟合精度
learning_rate = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# 初始化动量 (Momentum) 和 RMSprop 缓存
mW1, vW1 = np.zeros_like(W1), np.zeros_like(W1)
mb1, vb1 = np.zeros_like(b1), np.zeros_like(b1)
mW2, vW2 = np.zeros_like(W2), np.zeros_like(W2)
mb2, vb2 = np.zeros_like(b2), np.zeros_like(b2)

# ==========================================
# 4. 模型训练 (反向传播与梯度下降)
# ==========================================

epochs = 5000  # 迭代次数
m_train = X_train.shape[0]

print("开始训练...")
for epoch in range(1, epochs + 1):
    # --- 前向传播 (Forward Propagation) ---
    Z1 = np.dot(X_train, W1) + b1
    A1 = np.maximum(0, Z1)           # 隐藏层 ReLU 激活
    Z2 = np.dot(A1, W2) + b2         
    Y_pred = Z2                      # 输出层 (回归任务，线性输出)

    # --- 计算损失 (MSE Loss) ---
    loss = np.mean((Y_pred - Y_train) ** 2)
    
    # --- 反向传播 (Backward Propagation) ---
    # 输出层梯度
    dZ2 = 2 * (Y_pred - Y_train) / m_train
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    # 隐藏层梯度
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1.copy()
    dZ1[Z1 <= 0] = 0                 # ReLU 导数：Z1 <= 0 时梯度为 0
    
    dW1 = np.dot(X_train.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # --- Adam 权重更新 ---
    t = epoch
    
    # 更新 W1
    mW1 = beta1 * mW1 + (1 - beta1) * dW1
    vW1 = beta2 * vW1 + (1 - beta2) * (dW1 ** 2)
    mW1_hat = mW1 / (1 - beta1 ** t)
    vW1_hat = vW1 / (1 - beta2 ** t)
    W1 -= learning_rate * mW1_hat / (np.sqrt(vW1_hat) + epsilon)
    
    # 更新 b1
    mb1 = beta1 * mb1 + (1 - beta1) * db1
    vb1 = beta2 * vb1 + (1 - beta2) * (db1 ** 2)
    mb1_hat = mb1 / (1 - beta1 ** t)
    vb1_hat = vb1 / (1 - beta2 ** t)
    b1 -= learning_rate * mb1_hat / (np.sqrt(vb1_hat) + epsilon)
    
    # 更新 W2
    mW2 = beta1 * mW2 + (1 - beta1) * dW2
    vW2 = beta2 * vW2 + (1 - beta2) * (dW2 ** 2)
    mW2_hat = mW2 / (1 - beta1 ** t)
    vW2_hat = vW2 / (1 - beta2 ** t)
    W2 -= learning_rate * mW2_hat / (np.sqrt(vW2_hat) + epsilon)
    
    # 更新 b2
    mb2 = beta1 * mb2 + (1 - beta1) * db2
    vb2 = beta2 * vb2 + (1 - beta2) * (db2 ** 2)
    mb2_hat = mb2 / (1 - beta1 ** t)
    vb2_hat = vb2 / (1 - beta2 ** t)
    b2 -= learning_rate * mb2_hat / (np.sqrt(vb2_hat) + epsilon)
    
    # 打印训练进度
    if epoch % 500 == 0:
        print(f"Epoch {epoch}/{epochs} - Loss: {loss:.5f}")

# ==========================================
# 5. 模型测试与效果验证
# ==========================================

# 使用训练好的权重在测试集上进行前向传播
Z1_test = np.dot(X_test, W1) + b1
A1_test = np.maximum(0, Z1_test)
Y_test_pred = np.dot(A1_test, W2) + b2

test_loss = np.mean((Y_test_pred - Y_test) ** 2)
print(f"测试集完成！Test Loss (MSE): {test_loss:.5f}")

# ==========================================
# 6. 可视化作图与保存
# ==========================================
plt.figure(figsize=(10, 6))

# 绘制训练集散点图 (带有噪声)
plt.scatter(X_train, Y_train, color='gray', alpha=0.3, s=10, label='Training Data (with noise)')

# 绘制真实的底层函数曲线
plt.plot(X_test, Y_test, color='green', linestyle='--', linewidth=2, label='True Function f(x)')

# 绘制神经网络预测的曲线
plt.plot(X_test, Y_test_pred, color='red', linewidth=2, label='Neural Network Prediction')

plt.title("2-Layer ReLU Network Function Approximation (Pure NumPy)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)

# 动态获取当前代码文件所在的绝对目录
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, 'fit_relu_function.png')

# 将图片保存到代码同级目录
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"图片已成功保存至代码同级目录：\n{save_path}")

# 显示图表
plt.show()