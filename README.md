# REINFORCE Algorithm - CartPole Demo

Demo triển khai thuật toán REINFORCE (Policy Gradient) để giải quyết bài toán CartPole trong môi trường OpenAI Gym.

## Giới thiệu

REINFORCE là một thuật toán Policy Gradient cơ bản trong Reinforcement Learning. Thuật toán này học trực tiếp policy (chính sách) bằng cách tối ưu hóa expected return thông qua gradient ascent.

**Bài toán CartPole**: Cân bằng một cây gậy thẳng đứng trên một xe đẩy bằng cách di chuyển xe sang trái hoặc phải. Mục tiêu là giữ gậy không đổ trong thời gian dài nhất có thể.

## Cấu trúc dự án

```
.
├── CartPole.py    # Code chính chứa agent và training loop
└── README.md                # File này
```

## Yêu cầu

- Python 3.7+
- PyTorch
- Gymnasium (hoặc OpenAI Gym)
- NumPy
- Matplotlib (cho visualization)

## Cài đặt

```bash
pip install -r requirements.txt
```

Hoặc cài đặt thủ công:

```bash
pip install torch gymnasium numpy matplotlib
```

## Cách sử dụng

### Training

```bash
python reinforce_cartpole.py
```

### Tham số chính

Bạn có thể điều chỉnh các hyperparameters trong file code:

- `learning_rate`: Tốc độ học (mặc định: 0.01)
- `gamma`: Discount factor (mặc định: 0.99)
- `num_episodes`: Số episodes để train (mặc định: 1000)
- `hidden_size`: Kích thước hidden layer (mặc định: 128)

## Thuật toán REINFORCE

### Công thức chính

Policy gradient được tính theo công thức:

```
∇J(θ) = E[∑ ∇log π(a|s,θ) * G_t]
```

Trong đó:
- `θ`: Parameters của policy network
- `π(a|s,θ)`: Probability của action a tại state s
- `G_t`: Discounted return từ timestep t

### Quy trình training

1. Khởi tạo policy network với random weights
2. Với mỗi episode:
   - Thu thập trajectory (states, actions, rewards)
   - Tính discounted returns cho mỗi timestep
   - Cập nhật policy theo policy gradient
3. Lặp lại cho đến khi đạt được performance mong muốn

## Kết quả mong đợi

- **Episode đầu**: Reward trung bình ~20-50
- **Sau 200-300 episodes**: Reward đạt ~200 (solved threshold)
- **Convergence**: Agent có thể cân bằng gậy trong 200+ timesteps

## Ưu điểm và Hạn chế

### Ưu điểm
- Đơn giản, dễ implement
- Hoạt động tốt với action space rời rạc và liên tục
- Có thể học stochastic policies

### Hạn chế
- High variance trong gradient estimates
- Sample inefficient (cần nhiều episodes)
- Có thể hội tụ chậm

## Cải tiến có thể áp dụng

1. **Baseline**: Sử dụng value function để giảm variance
2. **Actor-Critic**: Kết hợp với value function approximation
3. **Advantage Function**: Sử dụng A(s,a) thay vì G_t
4. **Entropy Regularization**: Khuyến khích exploration

---

**Lưu ý**: Đây là phiên bản demo cho mục đích học tập. Để áp dụng vào production hoặc các bài toán phức tạp hơn, cần thêm nhiều optimization và engineering practices.
