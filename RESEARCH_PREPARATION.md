# 项目 G：private-RSI 研究准备记录

更新日期：2026-10-02（Asia/Shanghai）

## 来源与冻结状态

- 上游：<https://github.com/StephCpa/private-RSI>
- 默认分支：`claude/private-recursive-agent-evolution-384c6h`
- 已核对提交：`0e670e0bd2a561673df7a8ee2e4da2a82d1d71fd`
- 本地目录：`E:\科研工作流\项目G-private-RSI`
- 上游仓库状态：planning stage；没有 LLM 实验，所有性能数字在当前版本中都应视为假设或分析性校准结果。

## 文件地图

- `README.md`：项目定位、阅读顺序、可复现命令与四条分析性发现。
- `docs/01_review_of_gpt_analysis.md`：对早期 GPT 分析逐点复核，指出 DP-ES 信号弱、机制形状、数据规模和隐私攻击面等关键分歧。
- `docs/02_research_plan.md`：完整研究计划；定义 DP-RAE、Mode L/Mode G、MT-Ops、基线 B0–B10、假设 H0–H5、实验 E0–E8、统计、预算、时间表和门控。
- `docs/03_related_work.md`：截至 2026-10-02 的相关工作地图及核验状态。
- `docs/04_privacy_and_security_spec.md`：K1–K8 内核不变量、两个 DP 定理、记账规则、硬化和审计计划。
- `analysis/calibration.py`：自包含的 GDP/RDP/SVT/private-selection 校准分析。
- `results/calibration_output.md`：由脚本生成的数字输出。
- `tests/test_calibration.py`：16 个校准与机制测试。

## 当前已验证的内容

1. `python analysis/calibration.py` 可运行并生成校准输出。
2. `python -m unittest discover -s tests -v`：16/16 通过。
3. 校准显示：DP-ES 发布设置下的私有信号噪声标准差约为 1.0；在候选质量接近时，选中最佳提示只获得约 3% 的 oracle gain。这个结果支持先做 E0（ε=0 控制），但它仍是分析/模拟结果，不是 LLM 实验结果。
4. 在相同预算下，全批评分的分析性噪声标准差约 0.109；SVT/私有选择在“候选很多、接受升级少”的 RSI 形状下可能更有利。机制选择必须由 E1 replay 决定。
5. 5 个百分点的 meta-gain 在 ε=1 时大致需要 1,000–3,000 个用户，数据规模是主要约束。

## 研究主线

核心问题是：在用户级 `(ε, δ)` 隐私预算下，共享 improver 能否从多个租户的私有经验中学习并改进“改进程序”本身。计划把改进对象分为 task agent、improver、system/lineage 三层；主验证环境是新建的 MT-Ops 多租户工具使用沙箱，LaMP 用于 Mode L 外部验证，GSM8K 只承担 DP-ES 连续性控制。

首要门控顺序：

- G0：MT-Ops 的 oracle 相对 public-only 至少提升 15 pp，且 sandbox 信息流测试通过。
- G1：先证明非 DP meta-gain（至少 5 pp 且 CI 排除 0），再证明某种机制在 ε≤4、n≤2,000 的 replay 中保留至少 50% 的增益；Mode G 代理相关性不足时降级为辅助模式。
- G2：完整信息流审计、红队、基线复现通过。
- G3：在 ε≤4 的某个设置下 H0/H1a 成立，且 transplantation gain 在匹配预算后仍存在。

## 建议立即开展的 WP0

1. 固定环境、随机种子、模型版本、token 上限和数据生成器版本；把隐私计划参数写成不可变配置。
2. 先复现 E0：DP-ES、ε=0 public-only、同 ε 全批评分；每个条件 10 seeds，先只报告控制结果。
3. 实现 MT-Ops v0：租户边界、公共/私有规则、重叠参数 ω、可审计事件日志和非私有 oracle。
4. 实现 DP kernel 最小骨架：ledger、CSPRNG、sandbox、Gaussian/SVT、固定计划检查和完整 transcript 导出。
5. 以非私有 utility matrix 做 E1 replay，先筛机制与功效，再决定昂贵的 LLM 运行矩阵。
6. 为每个结果保留 claim/evidence 表：分析性校准、replay、真实 LLM 实验和外部论文基线分开记录。

## 研究边界

当前不能把仓库里的“约 3% oracle gain”“需要 1,000–3,000 用户”“定理 1/2”写成已被实验确认的系统性能；前两者是校准分析，定理依赖实现满足规范中的不变量。下一阶段的第一项可审查成果应是 E0 + MT-Ops G0，而不是直接运行 E3 主实验。

## WP0 已完成进展（2026-10-02）

已实现 `benchmarks/mtops/` 的 MT-Ops v0 确定性模拟器，并生成 `results/mtops_g0.json`、`results/mtops_g0.md` 和 `results/mtops_validity_sweep.json`。默认配置为 300 个 public、2,000 个 private、500 个 test 租户，四类任务、40 条共享规则和每租户一条专属规则。

G0 在 `overlap=0` 下通过：oracle 查询成功率 1.000，public-only 为 0.000，差值 1.000，超过 0.15 门槛。overlap 扫描中 public-only 成功率从 0.000（0.00）单调上升到 0.485（1.00）；这证明公共/私有重叠旋钮生效，但不构成 DP-RAE 或 LLM 效果证据。MT-Ops 测试 5/5 通过，原有校准测试 16/16 通过。

随后完成 E1 的零 LLM 成本 utility-matrix replay：60 个候选 × 600 个合成租户、1,000 次 replay，比较 release-all、简化 SVT 和 private-selection。默认矩阵下，`Q=60` 的 ranking accuracy 分别为 0.228、0.003、0.972；这些结果只用于机制实现优先级，不能替代真实 agent utility matrix。输出见 `results/e1_replay.md` 和 `results/e1_replay.json`。

已补上 DP kernel 最小闭环 `dprae/kernel/`：当前仅包含 gamma=1、add/remove、有界求和、basic composition 下的 Gaussian release-all 与 exponential winner-only。5/5 kernel 测试通过；审计边界见 `results/kernel_minimal_audit.md`。这一步不等同于 G2，也没有实现 SVT、Poisson 放大、生产级 RDP/PLD accountant、sandbox 隔离或信息流红队。

已完成 `experiments/kernel_mtops_smoke.py`：将 MT-Ops v0 的 public-only 与 local-adaptation 合成基线接入 kernel，Gaussian release-all 和 winner-only 均成功写入独立 ledger；结果见 `results/kernel_mtops_smoke.json`。该 smoke 只证明数据流和账本闭环，不是 LLM 或 DP-RAE utility 结果。

已在 `gpu-gfkd`（8×RTX A6000）上完成第一轮真实 agent 非 DP Mode L 校准：Qwen2.5-7B-Instruct，8 public 训练租户、12 private 训练租户、12 test 租户；private strategy 为 34/48，public strategy 为 29/48，差值 +10.42 pp。12 个租户配对 bootstrap 95% 区间为 [-8.33, 33.33] pp，包含 0，因此 G1 未通过；结果仅支持扩大样本和固定策略后的复验。原始结果见 `results/real_agent_calibration.json`，报告见 `results/real_agent_calibration.md`。

扩展到 24 public、24 private、48 test 租户后，private strategy 为 127/192，public strategy 为 139/192，差值 -6.25 pp；配对 bootstrap 95% 区间为 [-10.42, -2.08] pp。两轮方向相反，说明当前“由模型自由生成策略 + 一次性 A/B 判断”的校准不稳定，不能支持正的 private-meta gain。下一轮需预注册固定候选策略，并让评估执行策略实际描述的交互式工具流程，再重新检验 Mode L 差值。
