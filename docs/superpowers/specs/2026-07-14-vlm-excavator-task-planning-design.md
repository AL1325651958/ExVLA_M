# 轻量视觉语言模型挖机任务规划设计

## 目标

新建一个 GPU 部署的高层任务规划项目。轻量视觉语言模型（VLM）理解自然语言、RGB 图像和高程图，并输出可验证的结构化任务计划；既有 V11 继续作为纯视觉的低层四关节预测模型。

第一阶段仅建设可审阅的规划数据集，不训练或部署 VLM。

## 系统边界

```text
自然语言指令 + RGB 历史 + 高程图 + V11 结构化观测
                         ↓
               轻量 VLM（任务理解、规划）
                         ↓ JSON
                  规则与安全校验器
                         ↓
             V11 纯视觉低层关节预测 / 回放
```

VLM 不输出连续关节角度、不直接向执行器发控制命令。V11 的模型前向不接收或使用 qpos。第一版候选模型为 Qwen2.5-VL-3B-Instruct，后续可在同一 JSON 协议下替换。

## 规划协议

每个规划样本的目标输出为严格 JSON：

```json
{
  "phase": "dig",
  "target_region": "front_left",
  "constraints": ["bucket_near_terrain", "avoid_cab_collision"],
  "next_phase": "lift",
  "success_condition": "dig_stroke_complete",
  "confidence": 0.86
}
```

第一阶段阶段集合固定为：`observe`、`approach`、`lower`、`dig`、`lift`、`swing_to_dump`、`dump`、`return`、`stop`。目标区域以相机视角的 `front_left`、`front`、`front_right`、`left`、`right`、`unknown` 表示。安全校验器将在后续 VLM 阶段验证 schema、允许的阶段转移、置信度阈值与风险约束。

## 现有数据审计结论

当前数据目录有 30 个 episode：306、490、75 各 10 个。29 个 episode 的 RGB、高程图和 qpos 可读取且长度一致。一个 490 HDF5 文件损坏，读取时直接跳过；不生成或保存损坏文件清单。

HDF5 内不存在自然语言指令、任务阶段、目标区域、完成条件或人工语义标注。现有连续数据包括 RGB、高程图和四关节状态；75 还含 qvel、car_pos、dig_pos，490 还含 pressure、timestamps。不同型号的额外字段不统一，因此基础流程只能依赖所有型号共有的视觉和 qpos，型号特有信号仅用于增强候选标签。

真实训练数据的关节顺序为 `[Boom, Arm, Bucket, Swing]`。

## 第一阶段：半自动标注数据集

### 输入与切片

保留原始 HDF5 不变。在 `data/planning_dataset/` 写入新产物。每个有效 episode 按固定时间跨度或关键关节事件切为短片段，每段包含起始、中间、结束 RGB 和高程图、原始帧号、型号和关节轨迹摘要。

### 候选标签

根据 qpos 差分和可用 qvel 生成每关节的运动方向与幅值，再按规则生成阶段候选。高程变化用于增强 dig/lift 判断；490 的 pressure 与 75 的附加位置字段可提高各自型号的候选质量，但不得成为跨型号的必需条件。

候选标签始终带有 `source: heuristic` 和 `confidence`，不能作为 VLM 训练真值。

### 人工审核

为每个候选片段导出预览视频，包含 RGB、高程图、四关节曲线、候选阶段和时间范围。审核者确认或修改阶段、目标区域、约束、下一阶段和完成条件。确认后的样本标记为 `source: human_verified`，并导出 JSONL。

### 数据划分

训练、验证与测试按完整 episode 切分，并尽可能保留一个未见型号或未见作业场景作为泛化测试。禁止同一 episode 的相邻片段跨集合。

## 质量与验收

- 读取失败的 episode 直接跳过，终端只显示有效 episode 总数，不记录损坏文件路径。
- 原始 HDF5 永不修改。
- 导出 JSON 必须通过 schema 校验。
- VLM 微调只使用 `human_verified` 样本；其他样本仅进入待审核池或提示词实验。
- 第一阶段验收为：有效数据可稳定读取、候选片段可播放、JSON 合法、人工审核后的训练/验证集无 episode 泄漏。

## 非目标

- 第一阶段不微调 VLM、不接入真实执行器、不做边缘部署。
- 不让 VLM 直接预测四个连续关节角或绕过安全校验。
- 不实现 V12 光流或新的低层控制网络。
