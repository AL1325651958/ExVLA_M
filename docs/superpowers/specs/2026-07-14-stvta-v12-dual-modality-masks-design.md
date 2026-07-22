# Excavator-STVTA V12 双模态关节 Mask 设计

## 目标

在 V11 的纯视觉时序 Video-to-Action 基线上构建 Excavator-STVTA V12。V12 为每个关节分别生成 RGB mask 和高程 mask，并让两组 mask 实际门控动作预测；它不再使用“先融合 RGB/高程、再产生一组共享 mask”的路径。

V12 不包含语言模型、任务规划或直接执行器控制。关节输出顺序保持为 `[Boom, Arm, Bucket, Swing]`，预测输入视频窗口之后的下一时刻绝对关节角。

## 命名与兼容性

V12 的正式名称为 **Excavator-STVTA V12**（Excavator Spatio-Temporal Video-to-Action）。对外新模块、训练脚本、可视化脚本和输出目录使用 `stvta`，不使用 `yolo`。

V9–V11 的结构、脚本和 checkpoint 均不改动。V12 独立识别自己的 checkpoint；加载器仍保留旧 checkpoint 的兼容路径。V12 不要求、也不尝试直接加载 V11 参数到其双分支 mask/decoder 模块。

## V12 网络结构

```text
RGB video       -> RGB backbone -> RGB temporal mixer -> RGB masks 1..4 -> RGB joint features --+
                                                                                              |
                                                                                              +-> per-joint fusion -> action
                                                                                              |
Elevation video -> Elev backbone -> Elev temporal mixer -> Elev masks 1..4 -> Elev joint features+
```

两条分支从输入到关节特征保持隔离：

- RGB 和高程图各自拥有 adapter/backbone/feature-pyramid/grid projection。
- RGB 与高程帧差各自通过独立 motion adapter 编码；禁止使用 V11 的 6 通道混合帧差分支。
- 各分支分别执行同空间网格位置上的时间混合。
- 每个分支使用独立的 mask generator，产生 4 个与关节顺序绑定的 mask。
- 每个关节分别以其 RGB mask 解码 RGB joint feature，以其高程 mask 解码 Elev joint feature。

关节 `j` 的融合定义为：

```text
alpha_j = sigmoid(FusionGate(rgb_joint_j, elev_joint_j))
joint_j = alpha_j * rgb_joint_j + (1 - alpha_j) * elev_joint_j
```

`alpha_j` 是 RGB 信任度，高程信任度固定为 `1 - alpha_j`。融合后的 `joint_j` 进入该型号、该关节对应的动作头。RGB mask 只由 RGB token 生成，高程 mask 只由高程 token 生成；两类 mask 不应在生成前互相污染。

## 输出接口

正常 V12 前向保持三个主输出，便于沿用既有调用模式：

```text
action                 [B, 8]
average_modality_masks [B, 2, 4, G, G]
spatial_modality_masks [B, 2, 4, T, G, G]
```

模态维 `2` 固定为 `0=RGB`、`1=Elevation`。`return_diagnostics=True` 时追加：

```text
fusion_alpha           [B, 4]
```

`return_aux=True` 仅在训练时追加当前姿态辅助预测；若同时请求 auxiliary 和 diagnostics，输出顺序固定为 `action, average_masks, spatial_masks, pose_aux, fusion_alpha`。推理不读取 qpos，且不请求 pose auxiliary。

## 训练

V12 只使用现有 RGB、高程图、qpos/action 数据，不需要人工 mask 标注。总损失由下一时刻关节 Action 损失和训练期视觉姿态辅助损失组成。

为避免某一模态长期被忽略，训练时以低概率随机屏蔽整个 RGB 分支或整个高程分支。屏蔽只发生在训练期，验证与部署保留双模态输入。该增强不要求两类 mask 形状相同，也不对其施加伪造的空间一致性监督。

不使用水平翻转；翻转会改变挖机左右手性、作业方向和关节语义。RGB 与高程图的亮度/对比度增强仍按同一个片段的时间维保持一致。

## 可视化与诊断

V12 可视化为每个关节显示：RGB mask、Elevation mask 和 `alpha_j`。重点观察：

- RGB 或高程哪个模态提供了该关节的主要证据；
- 左侧背景捷径出现在哪个模态；
- Bucket mask 是否覆盖地形和接触作业区；
- Swing 是否依赖全局机身/场景参照。

可视化不把 mask 解释为机械结构分割；它表示该模态下对关节动作预测有贡献的空间门控权重。

## 验证要求

- 改变 qpos 不改变 action、RGB masks、高程 masks 或 fusion alpha。
- 固定高程图、改变 RGB 时，高程 masks 精确不变；反向亦然。
- 8 个 mask 的形状、模态维顺序、关节顺序和 alpha 值域 `[0, 1]` 正确。
- V9–V11 旧 checkpoint 继续加载、训练和可视化。
- 对 V11/V12 同时报告总体和逐关节 MAE、R²。
- 在 RGB 遮挡与高程遮挡的验证条件下分别报告性能变化，以检验双模态利用而非单模态塌缩。

## 非目标

- 不实现语言条件、VLM 规划、光流、真实执行器控制或边缘部署。
- 不修改原始 HDF5 数据。
- 不将 mask 当作像素级机械结构真值或要求 RGB/high-elevation mask 空间相同。
