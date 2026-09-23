# 多平台合并指南（MERGE.md）

> 何时用: 新平台的 SDPA 实现开发完成后，合入本目录。
> 当前已合并: ascend910（Triton）与 p800-kunlunxin（厂商委托）。
> 前置阅读: [PLATFORM.md](PLATFORM.md)（当前平台的绑定清单）。
> 原则: **黄金/测试/注册链是平台无关资产，直接复用；只有 kernel 本体按平台分文件**。

## 0. 先判定分叉程度（决定合并方式）

| 情形 | 特征 | 合并方式 |
|---|---|---|
| **参数级** | 只改常数: BLOCK 尺寸 / num_warps / exp 选择 / ieee 开关 | **不拆目录**——`_profile.py` 加平台字段，wrapper 按平台取配置 |
| **结构级** | kernel 主体逻辑不同: 不同 pass 结构 / split-K / 不同归约路径 | **拆 `kernel/backends/`**（见 §1） |

判定方法: 新实现写完后对照 PLATFORM.md §2 的 8 项绑定逐项标记。
**≥3 项互斥不同**（如 exp2 在新平台是正优化、tile 上限 128、dot 无
同 dtype 限制）→ 结构级。只有 1-2 项且是常数 → 参数级。

## 1. 结构级合并: 目录操作

```
sdpa/
├── reference.py              # 不动（fp32 语义，平台无关判卷标准）
├── goldendata/               # 不动（397 组黄金直接复用，见 §2 Step3）
├── register.py               # 不动（dispatch_key 已参数化）
├── _profile.py               # 加新平台 profile 条目
├── PLATFORM.md               # §2 改为"每平台一节"继续维护
├── kernel/
│   ├── triton_level.py       # 变薄 facade（保持 sdpa_triton 入口不变，见 §3）
│   └── backends/             # ★ 新建
│       ├── __init__.py       #   平台选择器（按 device.type 分发）
│       ├── ascend910.py      #   Triton 主体（函数名不变）
│       └── p800_kunlunxin.py #   厂商 efficient-attention 委托
├── test/                     # kernel_level.py 不动（36 case 换 profile 即跑）
└── reports/
    ├── perf_fp16.json        #   保留为 ascend910 数据（或改名加平台后缀）
    └── perf_fp16_<plat2>.json#   新平台数据
```

**兼容承诺**: `from kernel.triton_level import sdpa_triton` 与
`register.register_a1(...)` 的现有引用**一个都不改**——facade 转发，
误配平台在调用守卫处显式报错（机制见 PLATFORM.md §5）。

## 2. 七步合并流程

### Step 1 — 新实现文件（`kernel/backends/<platform2>.py`）

三件套照抄 ascend910 版模式:

```python
PLATFORM = "<platform2>"
SUPPORTED_DEVICE_TYPES = ("cuda",)        # 新平台的 torch device.type

def sdpa_triton(query, key, value, ...):  # 同签名，同名（有意，见 §3）
    if query.device.type not in SUPPORTED_DEVICE_TYPES:
        raise RuntimeError(...)           # 调用守卫，错误信息指向本文件
    ...
```

### Step 2 — PLATFORM.md 平台绑定逐项重验（不可跳过）

Ascend Triton 项按现成实验脚本重跑；若新平台选择 Route B / 厂商委托，
对应项改为验证 vendor API、精度、mask 语义、异常与注册 key：

| 绑定项 | 重验方法 |
|---|---|
| dot 同 dtype 限制 | 去掉 P-cast 编译试试；通过则删 |
| exp2 快慢 | `script/perf_variants.py` 的 V2 单项 A/B |
| tile 上限 | `script/perf_explore.py` 的 A 段扫描（BM/BN>64 是否可编译） |
| causal 截断 | 预期为正优化（机制平台无关），数值验证确认即可 |
| fp32 ieee | 黄金 extreme case 直接检验 |
| device 上下文 | 哨兵检查（确定性+敏感）抓静默 no-op |
| native shim | causal+mask 并存是否被接受 |
| 注册 key | `probes/debug_reg.py` 模式: 注册→call count→dispatch dump |

结果写进 PLATFORM.md 的新平台小节——**相反结论要显式写**（如
"exp2 正优化，已启用"，防止后来者拿 ascend 结论误用）。

### Step 3 — 黄金复用（零成本）

```bash
python3 script/check_accuracy.py --impl triton --device <新平台dev>
```

397 组黄金在 CPU 生成、双向互验，**不需要任何重建**。新平台实现
直接对同一判卷标准验证——这正是黄金先行设计的回报。若新平台全过，
精度结论直接可比；若个别 case 失败，失败模式本身就是平台缺陷线索。

### Step 4 — kernel 层 36 case 复跑

```bash
python3 test/kernel_level.py <新平台profile>
```

12 shape × 3 dtype + 哨兵，覆盖矩阵（尾块/GQA/mask/decode）平台无关。
只改 `_profile.py` 的 profile 参数。

### Step 5 — A1 注册（新平台 key）

```python
from register import register_a1
lib = register_a1("AutogradCUDA") # ascend910 用 "AutogradPrivateUse1"
```

注意新平台可能不需要 autograd.Function 包装（那是 torch_npu 的
AutogradPrivateUse1 不 redispatch 逼出来的），先试朴素注册，
`test/op_level.py` 的拦截+梯度检查会告诉你答案。P800 实测
`AutogradCUDA` 可拦截，但 schema 默认参数会被省略，需要 wrapper 补齐。

### Step 6 — 性能三方对照

```bash
python3 script/bench_perf.py --json-out reports/perf_fp16_<plat2>.json
```

新平台的"原生"参照换成该平台官方 SDPA（CUDA=flash/cudnn）。
**性能数字不跨平台引用**（PLATFORM.md §3）。

### Step 7 — 文档同步

- PLATFORM.md: §2 拆成 ascend910 / <plat2> 两节
- REPORT.md: 实现矩阵加行、关键数字表加列（或分表）
- reports/performance.md: 新平台小节
- README: 快速开始加新 profile 用法

## 3. 平台选择器（已落地）

```python
import importlib

_IMPL_BY_DEVICE = {
    "npu": "ascend910",
    "cuda": "p800_kunlunxin",  # backend 内部再校验 torch_xmlir
}

def get_impl(device_type: str):
    name = _IMPL_BY_DEVICE.get(device_type)
    if name is None:
        raise RuntimeError(
            f"sdpa 无 {device_type!r} 平台实现，已注册: "
            f"{list(_IMPL_BY_DEVICE)}（见 MERGING 指南/PLATFORM.md）")
    return importlib.import_module(f".{name}", __package__)
```

`kernel/triton_level.py` 退化为 facade:

```python
"""兼容 facade: sdpa_triton 按输入设备自动路由到平台实现。"""
from kernel.backends import get_impl
PLATFORM = "multi(ascend910,p800-kunlunxin)"
SUPPORTED_DEVICE_TYPES = ("npu", "cuda")

def sdpa_triton(q, k, v, *a, **kw):
    return get_impl(q.device.type).sdpa_triton(q, k, v, *a, **kw)
```

## 4. FlagGems 上游路径（若走贡献）

上游仓库**本身就是多平台结构**（`runtime/backend/_nvidia/_cambricon/
_kunlunxin/...` 各厂商目录并列，已实际存在），所以:

- ascend910 版 → `_ascend/`（含 SDPA 的 aten 接入修复）
- platform2 版 → `_<vendor2>/`，**不进** `_ascend`
- 共享部分（语义/测试思路）随 PR 描述与测试文件走
- 平台选择由 FlagGems 的 DeviceDetector/profile 体系承担，
  不是我们的 facade（上游已有该机制，避免重复）

## 5. 合并完成判据（checklist）

- [x] p800-kunlunxin: 黄金 397/397 + kernel 层 37/37 + op 层拦截/梯度绿
- [x] PLATFORM.md 有 p800-kunlunxin 小节，相反结论已显式标注
- [x] 旧引用零改动（`kernel.triton_level.sdpa_triton` 保留）
- [x] 误配平台触发调用守卫（`probes/guard_check.py p800-kunlunxin`）
- [x] 性能 JSON 按平台命名，报告不跨平台引用数字
