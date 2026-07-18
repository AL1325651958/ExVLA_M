# V17.1 四类自监督掩膜优化策略

V17.1 为大臂、小臂、铲斗和回转分别学习关节条件自监督掩膜。掩膜不依赖像素级人工标注，而是通过关节角预测损失与掩膜正则项联合优化。

| 关节 | 英文名称 | 注意力偏置 $\lambda_m$ | 残差增强 $\lambda_v$ | 掩膜策略 |
|---|---|---:|---:|---|
| 大臂 | Boom | 1.5 | 0.8 | 半局部注意力，兼顾大臂、机身和部分全局信息 |
| 小臂 | Arm | 2.0 | 1.0 | 较强局部注意力，重点提取小臂连杆信息 |
| 铲斗 | Bucket | 2.5 | 1.0 | 最强局部注意力，重点提取铲斗及末端作业区域 |
| 回转 | Swing | 0.0 | 0.5 | 自由全局软注意力，不强制限制注意力位置 |

## 注意力偏置

掩膜通过对注意力 logits 添加偏置，引导解码器关注掩膜高响应区域：

$$
A_j=
\operatorname{Softmax}
\left(
\frac{Q_jK^{\mathsf T}}{\sqrt{d}}
+
\lambda_{m,j}\log(M_j+\varepsilon)
\right).
$$

其中：

- $M_j$ 为第 $j$ 个关节的空间掩膜；
- $\lambda_{m,j}$ 控制掩膜对注意力位置的引导强度；
- $\lambda_{m,j}$ 越大，模型越倾向于关注掩膜高响应区域；
- $\lambda_{m,j}=0$ 表示不使用掩膜限制注意力位置。

## 残差增强

掩膜通过残差方式增强高响应区域的视觉特征：

$$
V'_j=
V\odot
\left(1+\lambda_{v,j}M_j\right).
$$

其中：

- $V$ 为原始视觉特征；
- $V'_j$ 为第 $j$ 个关节对应的掩膜增强特征；
- $\lambda_{v,j}$ 控制掩膜区域的特征增强幅度；
- 当 $M_j=0$ 时，$V'_j=V$，原始视觉信息仍被完整保留；
- 当 $M_j$ 较大时，对应区域按照 $1+\lambda_{v,j}M_j$ 的比例增强。

因此，残差增强只强化与关节预测相关的区域，不会像硬掩膜一样阻断基础视觉信息通路。

## 铲斗掩膜

### 功能描述

铲斗（Bucket）是挖掘机的末端执行机构，主要完成物料切入、挖掘、装载和卸载。其运动通常表现为铲斗轮廓、斗齿位置，以及铲斗与地面或物料接触区域的连续变化。

### 掩膜优化参数

铲斗采用四个关节中最强的局部注意力约束：

$$
\lambda_{m,\mathrm{Bucket}}=2.5,
\qquad
\lambda_{v,\mathrm{Bucket}}=1.0.
$$

较大的注意力偏置 $\lambda_{m,\mathrm{Bucket}}$ 使 Bucket decoder 重点读取铲斗及末端作业区域。残差增强参数 $\lambda_{v,\mathrm{Bucket}}$ 则强化掩膜高响应区域，同时保留完整的基础视觉信息。

### 双模态掩膜

模型分别从 RGB 视频和 LiDAR 高程图中生成铲斗掩膜：

- RGB 掩膜主要提取铲斗外观、轮廓、斗齿位置和运动变化；
- 高程图掩膜主要提取斗尖位置、作业接触区域和局部地形高度变化。

两个模态的掩膜通过软并集进行融合：

$$
M_{\mathrm{Bucket}}
=
1-
\left(1-M_{\mathrm{Bucket}}^{\mathrm{RGB}}\right)
\left(1-M_{\mathrm{Bucket}}^{\mathrm{Elev}}\right).
$$

融合后的 $M_{\mathrm{Bucket}}$ 同时保留两个模态中的高响应位置，并被展平后输入 Bucket cross-attention decoder。

### 注意力引导

铲斗联合掩膜通过注意力偏置引导 Bucket query：

$$
A_{\mathrm{Bucket}}
=
\operatorname{Softmax}
\left(
\frac{Q_{\mathrm{Bucket}}K^{\mathsf T}}{\sqrt{d}}
+
2.5\log\left(M_{\mathrm{Bucket}}+\varepsilon\right)
\right).
$$

同时，掩膜通过残差方式增强相关视觉特征：

$$
V'_{\mathrm{Bucket}}
=
V\odot\left(1+M_{\mathrm{Bucket}}\right).
$$

最终，Bucket decoder 根据掩膜增强后的视觉记忆预测下一帧铲斗关节角度，并以正弦—余弦形式输出：

$$
\hat{y}_{\mathrm{Bucket}}
=
\left[
\sin\left(\hat{\theta}_{\mathrm{Bucket}}\right),
\cos\left(\hat{\theta}_{\mathrm{Bucket}}\right)
\right].
$$
