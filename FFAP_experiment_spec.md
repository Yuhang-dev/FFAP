# Feature-Fidelity-Aware Pruning (FFAP) — 实验说明文档 (for Codex)

> **用途**:本文件交由 Codex（或其它编码 agent）逐步推进实验。每个任务都有明确的输入、输出、验收标准（DoD, Definition of Done）。请**严格按阶段顺序执行**,在每个 GATE 处停下来等待人类确认,不要跳过 Stage 2 的 kill/continue 闸门。
> **目标会议**:AAAI-27。Abstract due 2026-07-21,Full paper due 2026-07-28(AoE / UTC-12)。
> **硬件**:单卡 NVIDIA RTX PRO 6000,96GB VRAM。所有实验必须在单卡上可行。
> **不做的事**:不做推理加速/吞吐量;不把 speedup 当指标。

---

## 0. 项目核心论点(写代码时始终对齐这个目标)

一句话:**现有压缩(剪枝/量化)用 PPL/loss 当作"什么该保留"的代理,但它系统性漏掉了内部可解释 feature 的损伤。我们提出用「因果验证过的 SAE feature fidelity」作为剪枝/量化的 saliency 目标,在相同稀疏度下比 Wanda/SparseGPT 更好地保留下游能力,尤其是安全/对齐边界。**

关键的科学支点(必须在代码与实验里体现):
- **几何 feature 存活度 ≠ 因果重要性**(这是 Borobia et al. arXiv:2603.25325 的 RQ5 结论)。因此本项目的 saliency **必须基于因果加权的 feature 重要性**,而不是简单的几何存活度。Stage 2 的闸门就是为验证这一点设的。

四个贡献(代码与实验要分别支撑):
1. **Diagnostic**:给出超越 PPL 的损伤度量;证明几何存活度有误导性,因果加权 feature fidelity 才预测能力退化。
2. **Method(核心)**:feature-fidelity-aware saliency,注入 Wanda/SparseGPT 权重打分。
3. **Safety case study(最强卖点)**:压缩优先损伤 refusal/safety feature;本方法可选择性保护。
4. **Robustness(防守)**:多 seed、多 width、subspace 层面证明 saliency 信号稳定。

---

## 1. 环境与依赖

### 1.1 基础环境
- Python 3.10+,CUDA 对应 PyTorch 2.3+。
- 关键库:`torch`, `transformers`, `datasets`, `accelerate`, `safetensors`, `numpy`, `scipy`, `scikit-learn`, `tqdm`, `einops`。
- SAE 相关:`sae_lens`(EleutherAI/Joseph Bloom 系)用于加载 Gemma Scope / Llama Scope;`huggingface_hub` 下载权重。
- 评测:`lm-eval`(EleutherAI lm-evaluation-harness)用于 ARC/HellaSwag/MMLU 等。

### 1.2 第一周必须先做的环境验收(Task 0)
**Task 0 — 环境冒烟测试**
- DoD:
  - 能在 96GB 卡上加载 `google/gemma-2-2b`(bf16, ~5GB)并跑通一次前向。
  - 能用 `sae_lens` 加载一个 Gemma Scope SAE(release `gemma-scope-2b-pt-res`,任选一层,如 layer 12,width 16k),对一批激活做 encode/decode,打印 reconstruction MSE 与 L0。
  - 能跑通 `lm-eval` 对 dense gemma-2-2b 在 ARC-Easy 上的一次评测(小子集即可),拿到 baseline 数字。
- 产出:`logs/task0_smoke.json`(含模型加载耗时、显存峰值、SAE MSE/L0、ARC-Easy baseline acc)。

---

## 2. 模型与 SAE 资源清单(优先用现成的,不自训)

### 2.1 主力模型(base,用于通用能力实验)
- **Gemma-2-2B**(`google/gemma-2-2b`)+ **Gemma Scope** SAE(`google/gemma-scope-2b-pt-res`,提供全层 SAE,width 16k 与 65k/262k)。**主战场,实验闭环最短。**
- **Llama-3.1-8B-Base**(`meta-llama/Llama-3.1-8B`)+ **Llama Scope**(`fnlp/Llama-Scope`,256 个 SAE,每层 32K 与 128K features)。**用于规模复现(Stage 3 后期)。**

### 2.2 Instruct/Chat 模型(用于 safety case study)
按可行性优先级:
1. **Gemma-2-9B-IT** + Gemma Scope 的 instruction-tuned SAE(注意:官方只发布了 layer 9/20/31 的 IT SAE,够做 targeted case study,不够做全层 sweep)。
2. 备选:Goodfire / Neuronpedia 上的现成 instruct SAE(如 Qwen2.5-7B-IT SAE、Llama-3.3-70B-Instruct SAE——70B 仅用于 inference/分析)。
3. **base→instruct 迁移**:已有证据表明 base 模型 SAE 能较好重建 IT 模型激活(Gemma Scope 论文 §;gemma-2-2b-it unlearning 实验报告 chat 模型仅 +0.886 loss vs base +0.863)。所以**优先方案**:直接把 `gemma-2-2b` 的 Gemma Scope SAE 套到 `gemma-2-2b-it` 上,先验证迁移质量(见 Task 6)。
4. **fallback**:若现成 SAE 在目标 instruct 模型上重建质量不足,用 FAST recipe(arXiv:2506.07691)在单卡上训单层/少数几层 instruct SAE(天级,不是周级)。**仅在 Task 6 判定不达标时才启动。**

### 2.3 Safety 锚点(turnkey,无需训练)
- **Arditi et al. 单方向 refusal**(arXiv:2406.11717):用现成方法在 instruct 模型上抽出 refusal direction,作为 SAE-derived refusal feature 的交叉验证 ground truth。

---

## 3. 数据集

- **Calibration / 激活采样**:C4 或 WikiText-2,128–512 条序列(标准 Wanda/SparseGPT 设置)。
- **PPL 评测**:WikiText-2 test。
- **通用能力评测**:ARC-Easy/Challenge、HellaSwag、MMLU(子集即可起步,后期补全)、可加 PIQA/WinoGrande。
- **Safety / refusal 评测**:
  - 有害指令集(如 AdvBench harmful behaviors 子集)→ 测 refusal rate。
  - 良性指令集 → 测 over-refusal(false refusal rate)。
  - 如有,接上你原来的 BCR / preference boundary 指标。
- **因果验证用探针数据**:针对每个待研究 feature/behavior,准备能激活该 feature 的正例与不激活的负例(用于 ablation/patching)。

---

## 4. 代码结构(建议 Codex 按此搭建)

```
ffap/
  configs/                # yaml 配置:模型、SAE、稀疏度、方法
  data/                   # 数据加载与 calibration 采样
    loaders.py
  models/
    load_model.py         # 加载 HF 模型,bf16,device_map
    load_sae.py           # 统一封装 Gemma Scope / Llama Scope SAE 加载
  pruning/
    base.py               # 剪枝基类:接口 prune(model, calib, sparsity, saliency_fn)
    magnitude.py          # baseline: 幅值剪枝
    wanda.py              # baseline: Wanda
    sparsegpt.py          # baseline: SparseGPT
    ffap.py               # 本方法: feature-fidelity-aware saliency
  features/
    extract.py            # 抽取 SAE feature 激活(给定 model + SAE + data)
    survival.py           # 几何存活度指标(Jaccard、firing-rate shift、decoder cosine drift)
    causal.py             # 因果重要性:ablation / activation patching → 每个 feature 的 causal score
    saliency.py           # 把 causal feature importance 映射成 weight-level S_feature(w_ij)
  eval/
    ppl.py
    capability.py         # 封装 lm-eval
    safety.py             # refusal / over-refusal / BCR
  robustness/
    multi_seed.py         # 多 seed / 多 width SAE 下的指标稳定性
    subspace.py           # subspace-level 指标(应对 SAE non-canonicity)
  experiments/            # 每个 Stage 的可复现脚本
    stage1_diagnostic.py
    stage2_gate_causal.py
    stage3_method.py
    stage4_robustness.py
  logs/  results/  figures/
```

### 4.1 关键接口约定
- `saliency_fn(weight, activation_stats, feature_stats) -> Tensor`:返回与 weight 同形状的"保留优先级"分数,越大越该保留。
- FFAP 的 saliency:`S = S_wanda + lambda * S_feature_causal`,其中 `S_feature_causal` 由 `features/saliency.py` 计算——衡量某权重对**因果上重要的 feature 方向**的写入/读出贡献。`lambda` 作为超参在 configs 里扫。

---

## 5. 分阶段实验计划(GATE 处必须停)

### Stage 1 — Diagnostic 复现 + 升级(Week 1)
**目的**:站在 Borobia/Duan 肩上,先复现"PPL 没坏但 feature 坏了",并把度量做扎实。**这是 motivation,不是主贡献,别在此过度投入。**

- **Task 1.1**:在 gemma-2-2b 上跑 magnitude / wanda / sparsegpt,稀疏度 {20,30,40,50,60}%,记录 PPL + ARC/HellaSwag。
- **Task 1.2**:对每个剪枝后模型,用固定 SAE(同一 release/width,**冻结 SAE**)抽 feature 激活,计算:active-feature Jaccard、firing-rate shift、decoder-direction cosine drift。
- **Task 1.3(关键图)**:画两条曲线——横轴 sparsity,纵轴分别为 (a) PPL 退化、(b) SAE feature damage。**目标:展示两者不一致**(某稀疏度下 PPL 升 <5% 但 feature damage 明显)。
- DoD:`figures/stage1_ppl_vs_featuredamage.png` + `results/stage1.csv`。复现出 Wanda 比 magnitude 更保 feature structure(对齐 Borobia 趋势,作为 sanity check)。

### Stage 2 — ★KILL/CONTINUE GATE★:因果重要性验证(Week 1–2)
**这是全项目最关键的一步。必须先过这关再写方法。**

- **Task 2.1**:为一组 feature 同时计算 (a) 几何存活度(Stage 1 的指标)、(b) **因果重要性**(对该 feature 做 ablation / activation patching,测下游能力或特定行为的变化)。
- **Task 2.2**:检验假设——**因果加权的 feature fidelity 是否比几何存活度、以及比 PPL,更能预测剪枝后的能力保留**?用相关性分析(Spearman)+ 预测实验。
- **★GATE 判定★**:
  - **PASS(继续)**:因果加权 feature fidelity 与能力保留显著相关,且优于几何存活度与 PPL 的预测力。→ 进入 Stage 3,saliency 用因果加权定义。
  - **FAIL(止损/转向)**:若因果加权也预测不了能力保留 → **不要硬做 method**。转向二选一:(i) 收缩成"诊断 + 量化"论文;(ii) 转 calibration 方向(见附录 B,但已知更卷)。
- DoD:`results/stage2_gate.json` 给出明确 PASS/FAIL + 相关性数字 + 一段人类可读结论。**停下来等人类确认再继续。**

### Stage 3 — 方法构建与评测(Week 2–4)
- **Task 3.1**:实现 `pruning/ffap.py`,saliency = `S_wanda + lambda * S_feature_causal`。扫 `lambda`。
- **Task 3.2**:在 gemma-2-2b 上对比 magnitude / wanda / sparsegpt / **FFAP**,稀疏度 {20..60}%,指标:PPL + 通用能力 + feature fidelity。**核心结论图:相同稀疏度下 FFAP 的能力保留 ≥ baseline 2–3 点,或同等能力下多剪 5–10%。**
- **Task 3.3(safety case study,最强卖点)**:
  - 用 instruct 模型(gemma-2-9b-it 或备选)+ instruct SAE。
  - 先验证:压缩是否**优先**损伤 refusal/safety feature(用 Arditi refusal direction 交叉验证这些 feature 的身份)。
  - 再验证:FFAP 能否选择性保护这组 feature → 在相同稀疏度下 refusal rate 退化更小、over-refusal 不恶化。
  - **基线必须包含 AAPP**(arXiv:2511.07482)做 refusal rate 正面对比,并在论文里讲清差异:FFAP 是 static、SAE-feature 层面、剪枝+量化通用;AAPP 是 dynamic、probe/circuit 层面、仅结构化剪枝。
- **Task 3.4**:规模复现——在 Llama-3.1-8B + Llama Scope 上重跑核心对比;并加一个量化设置(如 INT4/INT7),把 Duan 的量化发现统一进同一框架。
- DoD:`figures/stage3_method_capability.png`、`figures/stage3_safety.png`、`results/stage3_*.csv`。

### Stage 4 — Robustness 防守(Week 4–5,非可选)
- **Task 4.1**:在 ≥2 个 seed、≥2 个 SAE width 下重算 feature fidelity 指标,证明 FFAP 的 saliency 信号稳定。
- **Task 4.2**:把指标从"单个 feature"上升到 **subspace 层面**(应对 SAE non-canonicity,引用 arXiv:2502.04878 与 arXiv:2606.12138)。
- DoD:`figures/stage4_robustness.png` + 一段"对 seed/width 不敏感"的证据。

### Stage 5 — 出图与写作支撑(Week 5)
- 汇总所有 results 成论文级表格/图;生成可复现脚本与 README;整理 supplementary。
- 对齐 AAAI:7-18 锁标题/abstract → 7-21 交 abstract → 7-28 交全文 → 7-31 交补充材料与代码。

---

## 6. 评测指标汇总(统一口径,避免各 Stage 不一致)

| 类别 | 指标 | 工具 |
|---|---|---|
| 语言建模 | WikiText-2 PPL | `eval/ppl.py` |
| 通用能力 | ARC-E/C, HellaSwag, MMLU(子集→全量), PIQA/WinoGrande | `lm-eval` |
| Feature 几何 | active-feature Jaccard, firing-rate shift, decoder cosine drift, effective rank | `features/survival.py` |
| Feature 因果 | ablation/patching 下游影响 → per-feature causal score | `features/causal.py` |
| Safety | refusal rate, over-refusal(false refusal), BCR/preference boundary | `eval/safety.py` |
| Robustness | 跨 seed/width 指标方差, subspace 稳定性 | `robustness/` |

报告原则:能力**和**安全都要在**相同稀疏度**下对比;feature 指标默认**冻结 SAE**、并在 Stage 4 报多 seed/width。

---

## 7. 显存与算力预算(单卡 96GB)
- gemma-2-2b bf16 ~5GB;Llama-3.1-8B bf16 ~16GB;均远低于 96GB,SAE 字典再加几百 MB–数 GB。
- SparseGPT 需要 Hessian/激活缓存,8B 级别在 96GB 内可行;如紧张,分层处理 + 释放缓存。
- 70B 模型仅用于 inference/分析(safety SAE 备选),不做训练。
- ablation/patching 是前向级操作,成本低;主要时间花在 SAE 激活抽取与多 seed 重复。

---

## 8. 执行纪律(给 Codex 的硬性要求)
1. **按 Stage 顺序执行,Stage 2 GATE 处必须停,输出 PASS/FAIL 后等待人类确认。**
2. 每个 Task 完成后写 `logs/<task>.json`(配置、随机种子、显存峰值、耗时、关键数字)与一句话结论。
3. 所有实验固定并记录随机种子;SAE 默认冻结。
4. 任何"feature 重要性"默认指**因果加权**;若用几何存活度,必须显式标注并说明理由(Borobia RQ5 会被审稿人攻击)。
5. 不引入推理加速类指标。
6. 代码可复现:每个 figure 对应一个一键脚本。

---

## 附录 A — 必读/必引文献(写 related work 与防守用)
- Borobia et al. 2026, *How Pruning Reshapes Features* (arXiv:2603.25325) — motivation + RQ5 威胁(几何存活≠因果重要),**本方法的立论支点**。
- Duan 2026, *Perplexity Can Miss SAE Feature Damage Under Quantization* (arXiv:2606.03002) — 量化侧 motivation。
- AAPP 2025, *Alignment-Constrained Dynamic Pruning* (arXiv:2511.07482) — safety 基线,需正面比较并划清界限。
- Gemma Scope (arXiv:2408.05147)、Llama Scope (arXiv:2410.20526) — SAE 资源。
- Arditi et al. 2024, *Refusal is Mediated by a Single Direction* (arXiv:2406.11717) — refusal 锚点。
- Leask et al. 2025, *SAEs Do Not Find Canonical Units* (arXiv:2502.04878);*Unstable Features, Reproducible Subspaces* (arXiv:2606.12138) — robustness 防守依据。
- Wei et al. 2024, *Assessing the Brittleness of Safety Alignment via Pruning* (arXiv:2402.05162) — safety-aware compression 的经典动机。

## 附录 B — 备用方向(仅当 Stage 2 FAIL 才考虑)
- **Calibration-as-coverage**:把剪枝 calibration 数据选择做成 feature/subspace coverage。**已知更卷且部分被抢**(COVERCAL arXiv:2604.24008 已用 submodular set cover 做量化 calibration;ICLR 2025 *Beware of Calibration Data* arXiv:2410.17711 是基础)。仅作为退路,不作为主线。

## 附录 C — 风险与对策速查
| 风险 | 后果 | 对策 |
|---|---|---|
| 几何存活≠因果重要 | saliency 保护错东西 | Stage 2 GATE;saliency 用因果加权 |
| SAE non-canonical(seed/width 敏感) | 指标被质疑是 artifact | Stage 4 多 seed/width + subspace 指标 |
| Borobia/Duan 出 v2 或被人抢先做成 method | 主贡献被削弱 | 投稿前复查 arXiv;强化 safety+量化+因果三个差异轴 |
| instruct 模型无现成高质量 SAE | safety case study 受阻 | 先验证 base→instruct 迁移(Task 6);不行再用 FAST 训单层(天级) |
| FFAP 打不过 Wanda | method paper 降级 | 退为诊断论文(已有 Stage 1/2 深度分析 + 因果发现支撑) |
