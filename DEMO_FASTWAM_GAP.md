# ADG-WAM Demo 与 FastWAM 仓库差距清单（Gap Analysis）

> 目标：评估“在现有 FastWAM 仓库上直接改造 ADG-WAM”是否可行，并给出最小落地路径。

## 结论（先看这个）

可以直接在本仓库做改造，**但不是小改几行就能跑通**。原因是你提出的 ADG-WAM 比 FastWAM 多了三类新能力：

1. Dream 使用/长度自适应决策（Usage + Horizon Gate）
2. 轻量 dream 生成（Residual 或 Keyframe）
3. Dream 可信度评估与融合（Dream Critic/Trust Gate）

FastWAM 现有代码对「视频-动作联合建模」非常完整，但对「按需触发 dream + dream 质量监督」还是空白。因此建议按“最小可行版本（MVP）→ 完整版”两阶段推进。

---

## 一、和当前仓库对齐的部分（已有能力）

- 有稳定的 world-action 主干：`FastWAM / FastWAMJoint / FastWAMIDM`。
- 有视频专家 + 动作专家 + MoT 融合结构，可承载你新增分支。
- 有 Hydra 配置体系，方便新增模型开关与损失系数。
- 有训练、评估、数据处理、文本 embedding 预计算等完整工程链路。

=> 这意味着你的方法**可以以“新增变体模型”方式接入，而无需推翻原仓库**。

---

## 二、核心 Gap（Demo 到可训练代码之间）

## G1. 缺少 Gate 模块（是否 dream + dream horizon）

Demo 里需要预测：
- `g_t`：当前样本是否触发 dream
- `T_t`：dream horizon（离散集合如 `{0,2,4,8,16}`）

仓库现状：
- 没有对应模块与监督目标；训练流程也没有 horizon 分类头。

需要新增：
- `DreamUsageHorizonGate`（MLP/Transformer 小头）
- 训练时的 gate loss（BCE + CE 或弱监督版本）
- 推理时阈值策略与离散 horizon 决策

---

## G2. 缺少轻量 Dream Generator

Demo 要求两种 dream：
- Residual Dream：预测 latent 残差
- Keyframe Dream：预测稀疏关键 latent

仓库现状：
- 只有 FastWAM 主干的视频/动作分支；没有独立 “future summary generator”。

需要新增：
- `ResidualDreamHead`
- `KeyframeDreamHead`
- 与主干 latent 的接口（输入 z_t，输出 dream summary）

---

## G3. 缺少 Dream Critic（可信度估计）

Demo 要求：
- 对 dream 打分 `c_t`
- 控制 dream 融合强度

仓库现状：
- 无 critic 分支、无 dream 打分数据、无对应损失项。

需要新增：
- `DreamCritic`
- 融合函数 `Phi(z_t, d_hat, c_t)`（建议 gated residual）
- critic 蒸馏训练入口（若使用离线教师分数）

---

## G4. 缺少 dream 监督数据/标签链路

Demo 有 teacher score（vis/logic/help）与 dream 标签。

仓库现状：
- 数据处理以行为数据、图像、文本 embedding 为主；没有 dream 监督字段。

需要新增：
- 数据字段：`dream_score`、`dream_target`、`horizon_label`（可选）
- 离线标注脚本（可先 stub，后接入 VLM 教师）
- dataset processor 对新字段的读取和 batch 组织

---

## G5. 缺少联合损失与训练调度

Demo 总损失：
\[
\mathcal{L}=\mathcal{L}_{act}+\lambda_{vid}\mathcal{L}_{vid}+\lambda_{gate}\mathcal{L}_{gate}+\lambda_{dream}\mathcal{L}_{dream}+\lambda_{crit}\mathcal{L}_{crit}
\]

仓库现状：
- 已有 video/action loss 体系；没有 gate/dream/critic 三类损失。

需要新增：
- 训练循环里扩展 loss dict
- 配置项里新增 `lambda_gate/lambda_dream/lambda_crit`
- 日志与 checkpoint 兼容（避免破坏现有流程）

---

## G6. 缺少“按需 dream”的推理路径

Demo 要求：
- 默认接近 FastWAM 单次前向
- 仅当 gate 触发时跑 dream 分支

仓库现状：
- `infer_action/infer_joint` 是固定流程，没有条件分支和预算控制指标。

需要新增：
- `if g_t < tau: bypass dream`
- `else: run dream + critic + fusion`
- 推理统计：触发率、平均 horizon、附加延迟

---

## 三、工程改造建议（最小可行路径）

## Phase 1（建议先做，2~4 周）

目标：先验证“自适应 gate + 轻量 residual dream”是否带来收益。

- 新增一个模型变体：`FastWAMADG`（只上 Residual Dream）
- Gate 先用弱监督策略（例如基于训练误差/任务类型伪标签）
- Critic 先用轻量回归头，不接大模型蒸馏
- 先在 LIBERO 子集做低数据实验（10/25/50/100%）

交付：
- 成功率 vs data fraction
- latency 与触发率
- 与 `fastwam`/`fastwam_joint` 对比

## Phase 2（完整版）

- 加 Keyframe Dream
- 加离线教师打分与 critic 蒸馏
- 完整消融（gate 粒度、dream 形式、是否 critic）
- 扩展到 RoboTwin 和长程任务

---

## 四、你这个想法是否“能直接改仓库验证”？

**可以，且非常适合从 FastWAM 出发。**

但建议你把“直接验证”定义为：
- 先做 MVP（Residual + Gate + 简易 Critic）
- 复用现有 FastWAM 训练/评估基础设施
- 不要第一版就把完整 VLM 教师链路、Keyframe、全量消融一次性全上

一句话：**方向是对的，工程上可落地，关键在于分阶段收敛。**
