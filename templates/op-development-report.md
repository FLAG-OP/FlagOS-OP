# 算子开发报告: &lt;算子名&gt;

| 项 | 值 |
|---|---|
| 算子名称 | `<op_name>` |
| 实现路线 | A1 torch 算子替换 / A2 FlagOS 融合算子 / B 厂商算子注册（勾选） |
| 开发层级 | torch 级 / Triton 级 / 硬件级（[定义](https://github.com/TruNcat3/FlagOS-OP/blob/main/docs/architecture.md#levels)） |
| 目标设备 | `<device profile 名>` |
| 开发者 | `<姓名>` |
| 日期 | `<YYYY-MM-DD>` |
| 报告状态 | 草稿 / 评审中 / 定稿 |

---

## 1. 环境配置

> 由 `scripts/gen_report_scaffold.py` 自动生成快照；手工补充项标注 `[手填]`。

### 1.1 硬件

| 项 | 值 |
|---|---|
| 设备型号 | `<如: Kunlunxin P800 OAM 96GiB ×2 / NVIDIA A100 ...>` |
| 拓扑 | `<PCIe/NVLink/...; 卡号映射（visible→物理）>` |
| 驱动/运行时版本 | `<xpu-smi/nvidia-smi 报告>` |

### 1.2 软件栈

| 组件 | 版本 |
|---|---|
| OS / 内核 | `<自动>` |
| PyTorch | `<自动>` |
| vLLM | `<自动>` |
| FlagGems | `<自动>` |
| vllm-plugin-FL | `<自动>` |
| Triton | `<自动>` |
| transformers | `<自动>` |
| 厂商库 | `<手填: 如 xtorch_ops / torch_npu 版本>` |

### 1.3 设备 profile 与关键环境变量

```yaml
# 自动粘贴 profile 内容
```

```bash
# 关键环境变量（自动）
```

### 1.4 环境特殊性说明 `[手填]`

<如: 容器限制、共享设备、已知怪癖开关（quirks）及其原因>

---

## 2. 算子定义

### 2.1 语义

<一段话描述算子做什么；数学公式或伪代码>

```
out = f(x, gate, ...)
```

### 2.2 PyTorch 参考实现

```python
def reference(x, ...):
    ...
```

### 2.3 接口签名

| 形态 | 签名 |
|---|---|
| aten（若 A1） | `aten::xxx(Tensor self, ...) -> Tensor` |
| dispatch（若 A2/B） | `call_op("<op_name>", obj, ...)` |
| 厂商 kernel（若 B） | `<pkg>.<fn>(x, out?) -> ...` |

### 2.4 数值规格

| 项 | 值 |
|---|---|
| 支持 dtype | bf16 / fp16 / fp32 |
| 内部计算精度 | fp32 累加 |
| 容差 | fp32: 1e-5；bf16/fp16: 1e-2（.float() 后比） |

---

## 3. 实现说明

### 3.1 路线选择理由

<为什么选这条路线: 算子类型（aten 已有/融合/厂商专用）、
调用方需求、性能目标>

### 3.2 kernel 实现要点

- 代码位置: `routes/<...>` / `examples/<...>`
- 关键技术: `<如 pointwise_dynamic 自动分块 / fp32 中间精度 / 厂商 API 调用>`

### 3.3 注册与分发

<注册方式、impl_id、优先级、policy 钉选方式（PER_OP 等）>

---

## 4. 验证结果

> 9 格矩阵中与本算子相关的格子 + 跨层一致性 + 黄金回归。
> `gen_report_scaffold.py` 自动填状态与耗时，详细数据手填。

| 物理栈层 | 验证层级 | 状态 | 耗时 | 关键结论 `[手填]` |
|---|---|---|---|---|
| 算子库层 | kernel 直测 | `<自动>` | `<自动>` | <精度 N/N、哨兵、性能> |
| 框架层 | op 注册/分发 | `<自动>` | `<自动>` | <注册数、策略切换> |
| 应用层 | framework 真实推理 | `<自动>` | `<自动>` | <调用次数、输出比对> |
| 跨层 | 算子库层↔框架层一致性 | `<自动>` | `<自动>` | <一致矩阵摘要> |
| — | 黄金回归 | `<自动>` | — | <prefix 匹配> |

### 4.1 kernel 层明细

| shape | dtype | vs 参考 max_err | 哨兵 | 性能 |
|---|---|---|---|---|
| | | | | |

### 4.2 framework 层明细

- 调用计数: `<N 次（pid 分片汇总）>`
- 输出比对模式: exact-per-prompt / first-2-token-rate
- 结果: `<摘要>`

---

## 5. 性能

| 实现 | 耗时/call | 有效带宽 | 相对参考加速 |
|---|---|---|---|
| 本实现 | | | |
| PyTorch 参考 | | | 1x |
| 厂商原实现（若有） | | | |

> 基准方法: 短采样（≤100 次）+ synchronize（[分配器陷阱](https://github.com/TruNcat3/FlagOS-OP/blob/main/docs/known-issues.md#10-性能基准的分配器陷阱)）。

---

## 6. 已知问题与风险

| # | 问题 | 影响 | 缓解 | 状态 |
|---|---|---|---|---|
| 1 | | | | |

<如涉及设备栈固有问题，链接 known-issues.md 对应条目>

---

## 7. 结论与后续

### 结论

<一段话: 是否达到验收标准>

### 后续工作

- [ ]

---

## 附录 A: 复现命令

```bash
python3 run.py --route <X> --level kernel --device <profile>   # 算子库层
python3 run.py --route <X> --level op --device <profile>       # 框架层
python3 run.py --route <X> --level framework --device <profile> # 应用层
python3 run.py --consistency --device <profile>                # 算子库层↔框架层
```

## 附录 B: 相关产物

| 产物 | 路径 |
|---|---|
| 结果 JSON | `results/<device>_<route>_<level>.json` |
| 黄金 | `golden/<device>_golden.json` |
| 代码 | `routes/... / examples/...` |
