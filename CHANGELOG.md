# YOLO-ST-VLA 版本演进记录

> 模型文件: `vla_model/model_yolo.py` | 训练脚本: `vla_model/train_yolo.py`

---

## V1 — 初始 YOLO 架构

**commit**: `bceeb93` 及之前

**架构**:
```
RGB+Elev → 双塔CSPDarknet → FPN-PAN → SpatialGridHead → grid tokens
  → Encoder(6层) → mask_head(softmax, τ=0.1) → gate → Decoder(3层) → delta_head → [B,8] sin/cos
```

**关键特征**:
- `delta_head`：预测关节角度变化量 Δqpos
- `mask_head` 在 **encoder 之后**：mask 基于全局混合后的 memory 打分
- 掩膜激活：`softmax(score / 0.1)` — 每个位置必须被某个 region 选中（无背景概念）
- 门控：`gate = Σmasks.clamp(0, 1)` — 多 mask 叠加后硬截断
- 稀疏性 loss：`-0.001 × entropy`（**符号反了**，实际奖励均匀分布）
- 多样性 loss：`0.01 × overlap`
- 共享 `action_head`、共享 `qpos_mod`
- `use_sincos_output=False`
- 无预训练 backbone

**问题**:
- 训练从随机初始化开始，epoch 1 MAE ~0.47
- 掩膜均匀覆盖整个画面
- 不同挖机类型泛化差

---

## V2 — 预训练 + 因果门控

**commit**: `a765ed2` (causal mask gating)

**新增**:
- YOLOv5s backbone 预训练权重加载（`pretrain_backbone.py` 生成本地 stub 包）
- 掩膜门控作用于 decoder cross-attention 的 memory

**架构变化**:
```
memory → mask_head → gate → gated_memory → decoder
```

**训练对比**:
| 指标 | V1 (无预训练) | V2 (预训练) |
|---|---|---|
| Epoch 1 Val MAE | — | 0.338 |
| Epoch 10 Val MAE | ~0.47 | 0.116 |

**仍存在的问题**:
- 掩膜虽然聚焦但**位置固定不变**（纯位置偏差，和画面内容无关）
- encoder 先全局混合再 mask，每个 token 特征高度相似，mask_head 无法区分空间
- delta_head 命名与实际输出矛盾（实际预测绝对值）

---

## V3 — 掩膜前置 + 统一 target

**commits**: `aa17055` (rename delta→action), `cee613f` (unify qpos[1:]), `65f8f91` (mask BEFORE encoder)

**架构变化**:
```
tokens → mask_head → gate → encoder → gated_tokens → decoder
```
- `mask_head` 移到 **encoder 之前**：输入 token 保留空间特异性（CNN 特征 + pos_embed），mask 可根据内容选择
- gate 作用于 encoder 输入，encoder 只能处理选中区域 → 梯度推动 mask 聚焦

**接口重命名**:
| 旧名 | 新名 | 含义 |
|---|---|---|
| `delta_head` | `action_head` | 预测绝对关节角度 |
| `decode_delta()` | `decode_action()` | — |
| delta_gt_rad | action_gt_rad | — |

**数据集统一**:
- 不再使用 HDF5 内不一致的 `action` key
- 统一 target = `qpos[1:]`（下一帧绝对角度）

**仍存在的问题**:
- 掩膜仍为 softmax（无背景概念）
- 门控硬截断 clamps 精细区域
- 无因果时序掩码，未来帧信息泄露
- 掩膜收敛到固定位置就不动了

---

## V4 — sigmoid 掩膜 + 因果时序 + NaN 修复

**commits**: `402acc6` (causal+sigmoid), `7427e91` (NaN fix: max gate), `8ecb179` (diversity /N), `155bffa` (causal mask back)

**架构变化**:

### 掩膜激活: softmax → sigmoid
```
之前: masks = softmax(score / 0.1, dim=spatial)     # 强制竞争，无背景
现在: masks = sigmoid(score)                         # 独立激活，允许全0
```
- 位置可被多个 region 同时选中
- 位置可全不选（背景），不会强制分配

### 门控: clamp → max + floor
```
之前: gate = Σmasks.clamp(0, 1)      # 叠加后硬截断，小信号丢失
现在: gate = max(masks).clamp(0.02)  # 数值稳定，小背景信号保留
```

### 损失函数重构
```
之前: sparsity = -0.001×entropy           → 符号反了
      diversity = 0.02×overlap.sum         → 未归一化，130倍MSE
      temporal = 0 (无)

现在: sparsity = 0.01×masks.mean()         → L1 鼓励稀疏
      diversity = 0.5×(overlap/N)².sum     → 归一化，平方惩罚
      temporal = 0.02×|mask_t - mask_{t-1}| → 时序平滑
```

### 因果时序掩码
```python
token_time = arange(T×G²) // G²
causal_mask = (time[j] < time[i]) × (-1e9)    # 用 -1e9 而非 -inf（amp 安全）
memory = encoder(gated_tokens, mask=causal_mask)
```
- t₀ 只能看 t₀
- t₁ 能看 t₀+t₁
- ...
- t₇ 能看全部

### NaN 修复记录
| 问题 | 原因 | 修复 |
|---|---|---|
| loss=nan, epoch 1 | diversity loss = 103 (130× MSE) | `overlap / N` 归一化 |
| loss=nan | prod(1-p) 门控梯度爆炸 | 改为 max() |
| LayerNorm NaN | gate=0 导致全部 token 为零 | `gate.clamp(min=0.02)` |
| causal NaN | `-inf` 在 fp16 下 softmax 分母=0 | 改为 `-1e9` |

---

## V5 — Per-Excavator 输出头

**commit**: `05ae15a` (per-excavator action_heads + qpos_mods)

**问题分析**:
V4 训练曲线中 Arm/Bucket 关节 Train R² ~0.85 但 Val R² ~0.3-0.5，差距巨大 → **过拟合**

根本原因：75/306/490 三款挖机臂长不同，相同关节角度对应的末端位置差异很大。共享 `action_head` 无法区分"这是 75 的小臂 0.3 rad vs 这是 490 的小臂 0.3 rad"。`excv_embed` 只注入到 encoder tokens，action_head 的参数是共享的。

**修改**:
```
之前:
  pool [B,512] → action_head(共享) → action [B,8]
  qpos[-1]     → qpos_mod(共享)    → correction

现在:
  pool → action_heads[excv_id] → action    (每款挖机独立预测头)
  qpos → qpos_mods[excv_id]   → correction (独立调制头)
```

```python
self.action_heads = nn.ModuleList([
    nn.Sequential(512→256→128→8) for _ in range(4)     # 75, 306, 490, other
])
self.qpos_mods = nn.ModuleList([
    nn.Sequential(4→32→8) for _ in range(4)
])
```

Forward 时按 `excavator_id` 路由到对应 head。

---

## 各版本汇总

| 特性 | V1 | V2 | V3 | V4 | V5 |
|---|---|---|---|---|---|
| Backbone 预训练 | ✗ | ✓ | ✓ | ✓ | ✓ |
| Mask 位置 | encoder后 | encoder后 | **encoder前** | encoder前 | encoder前 |
| Mask 激活 | softmax | softmax | softmax | **sigmoid** | sigmoid |
| 门控 | Σ→clamp | Σ→clamp | Σ→clamp | **max→floor** | max→floor |
| 因果时序掩码 | ✗ | ✗ | ✗ | **✓** | ✓ |
| 稀疏性 loss | -entropy(反) | +entropy | +entropy | **L1 mean** | L1 mean |
| 多样性 loss | 0.01×overlap | 0.01×overlap | 0.01×overlap | **0.5×(ov/N)²** | 0.5×(ov/N)² |
| 时序平滑 loss | ✗ | ✗ | ✗ | **0.02×|Δt|** | 0.02×|Δt| |
| Action head | 共享 | 共享 | 共享 | 共享 | **per-excv** |
| qpos_mod | 共享 | 共享 | 共享 | 共享 | **per-excv** |
| Target | 混合(HDF5) | 混合 | **qpos[1:]** | qpos[1:] | qpos[1:] |
| 输出维度 | [B,8] | [B,8] | [B,8] | [B,8] | [B,8] |
| 返回值 | (act, mask) | (act, mask) | (act, mask) | (act, avg, T) | (act, avg, T) |

---

## V5 完整架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         输入 [B, T=8, 3, 224, 224]                    │
│                    RGB × 2 + Elevation × 2 (双塔)                     │
└──────────┬──────────────────────────────┬───────────────────────────┘
           │                              │
    ┌──────▼──────┐                ┌──────▼──────┐
    │ CSPDarknet  │                │ CSPDarknet  │    YOLOv5s 预训练
    │ stem→s1→s4  │                │ stem→s1→s4  │
    │ p3,p4,p5    │                │ p3,p4,p5    │
    └──────┬──────┘                └──────┬──────┘
           │                              │
    ┌──────▼──────┐                ┌──────▼──────┐
    │   FPN-PAN   │                │   FPN-PAN   │    多尺度融合 (共享)
    │  n3,n4,n5   │                │  n3,n4,n5   │
    └──────┬──────┘                └──────┬──────┘
           │                              │
    ┌──────▼──────┐                ┌──────▼──────┐
    │SpatialGrid  │                │SpatialGrid  │    多尺度→14×14 网格
    │ [196, 768]  │                │ [196, 768]  │
    └──────┬──────┘                └──────┬──────┘
           │                              │
           └──────────┬───────────────────┘
                      │ cat + grid_proj Linear(1536→512)
              ┌───────▼────────┐
              │ tokens         │   [B, 1568, 512]
              │ + pos_embed    │   可学习时间编码 + sin/cos 空间编码
              │ + excv_embed   │   每类挖机独立 bias
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │   mask_head    │   LN→256→GELU→4→sigmoid
              │ [B,4,T,14,14]  │   逐token独立激活，无竞争
              └───┬───────┬────┘
                  │       │
      返回用于loss │       └──→ gate = max(·).clamp(0.02) → gated_tokens
                  │                         │
                  │              ┌──────────▼──────────┐
                  │              │   causal_mask       │   t 只能 attend ≤t
                  │              │   Encoder (4层8头)   │   norm_first, GELU
                  │              │   memory [1568,512]  │
                  │              └──────────┬──────────┘
                  │                         │
                  │              ┌──────────▼──────────┐
                  │              │   Decoder (2层)      │   4个可学习queries
                  │              │   cross-attn→memory  │
                  │              │   pool → [B,512]     │
                  │              └──────────┬──────────┘
                  │                         │
                  │              ┌──────────▼──────────┐
                  │              │action_heads[excv_id]│   每款挖机独立MLP
                  │              │   512→256→128→8     │   75/306/490/other
                  │              │   action [B,8]      │   sin/cos × 4关节
                  │              │ + qpos_mods[excv_id]│   训练时辅助调制
                  │              └──────────┬──────────┘
                  │                         │
                  ▼                         ▼
          masks_spatial              action [B,8]
        [B,4,T,14,14]               (sin/cos编码)
```

**损失**: MSE ×1.0 + L1稀疏×0.01 + 归一化重叠²×0.5 + 时序平滑×0.02

**参数量**: ~37.3M (hidden_dim=512)

---

## V6 — Per-Joint 结构化查询（无 Pooling）

**commit**: `fc82931`

**改动**:
- 取消 decoder 输出的 pooling：`decoded [B, 4, D]` → 4 个独立 joint feature
- 每个 joint 独立一个 `action_head[excv][joint]`，共 4×4=16 个微型 MLP
- `action_heads[excv]` 从单层列表变为嵌套 `ModuleList[ModuleList]`
- `joint_queries` 替换 `query_tokens`
- 软联合门控 `prod()` 替换 `max()`

**问题**: sin/cos 编码顺序不一致 → loss NaN → 回退到 `use_sincos_output=False`

---

## V7 — Joint-aware Spatial Action Mask（联合感知空间动作掩膜）

**commit**: `8810261`

**核心创新**: 每个关节掩膜独立门控该关节的 decoder 输入

```
masks [B, 4, N]
         │
  ┌──────┼──────┬──────┐
  ▼      ▼      ▼      ▼
memory⊙mask_0  memory⊙mask_1  memory⊙mask_2  memory⊙mask_3
  │      │      │      │
Query_0 Query_1 Query_2 Query_3   (各自独立的 decoder 视角)
  │      │      │      │
Head[excv][0]  Head[excv][1]  Head[excv][2]  Head[excv][3]
  │      │      │      │
Boom   Arm   Bucket  Swing
```

Mask → Feature → Joint → Action 闭环约束：每个 mask_j 是关节 j 的 decoder **唯一的特征入口**，掩膜聚焦错误则关节预测崩溃。

**数据集改进** (`75c940b`):
- 训练/验证按挖机类型分层划分，每款挖机都在 val 中出现
- 修复旧的连续索引划分（val 全是 75）导致的评估虚高

**参数量**: 39.7M (4×4 action heads)
**当前状态**: 训练中，epoch 3 Train R² 0.74，Val R² 0.52
