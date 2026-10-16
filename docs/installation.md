# 安装与 FlagGems 配合使用

[← 返回文档中心](index.md)

这份指南回答四个问题：

1. 下载 FlagOS-OP 后如何让 Python 找到它；
2. FlagOS-OP 算子和 FlagGems 如何分工、如何共存；
3. 如何注册算子、确认命中，并用子进程隔离实验。
4. 常见业务链路应选择 direct、FlagGems 还是 A1 映射。

> 当前发布方式：FlagOS-OP 以 **源码仓库** 交付，还没有发布 PyPI 包，
> 也没有提供 `pip install flagos-op`。不要把本仓库安装进 FlagGems 源码树；
> 两者是并列组件，通过 Python import 和 PyTorch dispatcher 配合。

## 1. 安装源码仓库

### 1.1 克隆仓库

```bash
git clone https://github.com/FLAG-OP/FlagOS-OP.git
cd FlagOS-OP
```

后续命令用下面的变量表示仓库根目录：

```bash
export FLAGOS_OP_ROOT="$PWD"
```

### 1.2 让 Python 找到仓库

FlagOS-OP 的包入口在仓库根目录，例如 `ops.sdpa_math`。在仓库外使用时，
把仓库根目录加入 `PYTHONPATH`：

```bash
export PYTHONPATH="$FLAGOS_OP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

只想临时运行一个命令时：

```bash
PYTHONPATH="$FLAGOS_OP_ROOT" python3 your_script.py
```

如果始终在仓库根目录执行命令，当前目录本身已在 `sys.path` 中，可以不设
`PYTHONPATH`。应用进程从其他目录启动时，推荐显式设置，避免“在仓库里能
import、部署环境不能 import”。

### 1.3 当前不是 editable package

当前仓库没有 `pyproject.toml` / `setup.py`，因此：

```bash
pip install -e .
```

不是受支持的安装方式。

这份仓库是一个源码算子库和验证矩阵：每个算子目录自带实现、测试、golden、
脚本和报告。当前使用方式以 `PYTHONPATH` 为安装边界。

## 2. 准备 FlagGems

### 2.1 优先使用已锁定运行时

厂商环境通常已经包含匹配版本的 FlagGems、Triton、torch 扩展和厂商
runtime。优先使用团队提供的共享镜像或容器。

检查当前环境：

```bash
python3 - <<'PY'
import torch
import flag_gems

print("torch:", torch.__version__)
print("flag_gems:", getattr(flag_gems, "__version__", "unknown"))
print("cuda_available:", torch.cuda.is_available())
PY
```

有环境锁文件的平台继续核对：

```bash
python3 scripts/check_env.py --device p800-kunlunxin
```

已登记的 Ascend、Cambricon、Kunlunxin 实测环境见
[软硬件环境版本汇总](environments.md)。

P800 锁定文件明确记录：FlagGems 和 `vllm-plugin-fl` 来自共享镜像本地
路径，不建议随意用 pip 覆盖安装。

### 2.2 开发者源码方式

如果需要修改 FlagGems，先按 FlagGems 上游或团队 fork 的说明安装依赖，
再让 Python 同时看到两个源码树：

```bash
# 示例；请替换为团队批准的 FlagGems fork / branch。
git clone https://github.com/FlagOpen/FlagGems.git
export FLAGGEMS_ROOT="/path/to/FlagGems"

export PYTHONPATH="$FLAGGEMS_ROOT/src:$FLAGOS_OP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

如果 FlagGems 已经通过上游推荐方式安装到当前环境，不需要再把它加入
`PYTHONPATH`。

## 3. 验证安装

```bash
python3 - <<'PY'
import torch
import flag_gems

from ops.sdpa_math import register_a1
from ops.sdpa import register_a1 as register_sdpa
from ops.embedding import register_a1 as register_embedding

print("FlagOS-OP import: OK")
print("torch:", torch.__version__)
print("FlagGems:", getattr(flag_gems, "__version__", "unknown"))
print("operator APIs: OK")
PY
```

成功输出应包含：

```text
FlagOS-OP import: OK
operator APIs: OK
```

再跑一个算子的一键验证：

```bash
cd ops/sdpa_math
python3 example.py p800-kunlunxin
cd ../..
```

## 4. FlagOS-OP 和 FlagGems 的关系

FlagGems 是 kernel 集合，主要有两种接入形态：

| 形态 | 机制 | 示例 |
|---|---|---|
| 标准 ATen 算子 | `flag_gems.enable()` 注册到 PyTorch dispatcher | GELU / softmax |
| vLLM 融合算子 | vLLM-FL 的 FlagGems backend / `call_op` | 融合激活类算子 |

FlagOS-OP 的算子通过自己的 `register_a1()` 或平台 facade 接入：

| FlagOS-OP 部分 | 接入点 | 与 FlagGems 的关系 |
|---|---|---|
| `ops/sdpa_math` | private ATen op A1 | FlagGems 通常没有同语义双输出实现 |
| `ops/sdpa` | 标准 SDPA / 平台 facade | 可能与 FlagGems 或厂商 backend 竞争 |
| `ops/embedding` | 标准 embedding A1 | 需避免与 FlagGems embedding 冲突 |
| 基础 ATen 算子族 | MLU A1 | MLU 三层与注册契约复验；当前应用证据为 C1 等价 torch API，C2 真实 vLLM 按环境补验 |

正确关系是并列组件：

```text
业务进程
  ├─ FlagGems kernel / FlagGems backend
  └─ FlagOS-OP ops.<operator>.register_a1 或平台 facade
       └─ PyTorch dispatcher / vendor runtime
```

## 5. 基本使用

### 5.1 查看支持范围

入口：

```text
ops/catalog.yaml
ops/README.md
```

`ops/catalog.yaml` 是算子 / 平台矩阵的事实来源。`planned` 只表示目标
范围，不表示已经支持。

### 5.2 运行算子示例

```bash
python3 ops/sdpa_math/example.py p800-kunlunxin
python3 ops/embedding/example.py p800-kunlunxin
python3 ops/clone/example.py cambricon
```

### 5.3 运行单个测试层

```bash
cd ops/sdpa_math
python3 test/kernel_level.py p800-kunlunxin
python3 test/op_level.py p800-kunlunxin
python3 test/framework_level.py p800-kunlunxin
cd ../..
```

## 6. 在应用代码中配合 FlagGems

下面是最小模式：FlagGems 负责周边 GELU，FlagOS-OP 负责 `sdpa_math`
private 双输出算子。

```python
import torch
import flag_gems

from ops.sdpa_math import register_a1

calls = {"n": 0}

# 共享运行时中推荐 scoped 启用，避免 full enable 覆盖大量 ATen 算子。
flag_gems.only_enable(include=["gelu"])

# 注册目标算子。返回对象必须保存引用，否则可能被 GC。
libs = register_a1(
    dispatch_key="AutogradCUDA",
    counter=calls,
    impl="p800",
)
assert libs

# Replace cuda:1 and p800 with the device/profile in your runtime. CPU smoke
# tests can use register_a1("CPU", impl="torch").
q = torch.randn(1, 4, 128, 64, device="cuda:1", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

with torch.no_grad():
    output, probabilities = (
        torch.ops.aten._scaled_dot_product_attention_math(
            q, k, v, None, 0.0, True, None, enable_gqa=False
        )
    )
    activated = torch.nn.functional.gelu(output)

print(output.shape, probabilities.shape, activated.shape)
assert calls["n"] == 1
```

当前 P800 运行时在启用 FlagGems GELU 时会打印一次
`Overriding a previously registered kernel ... aten::gelu` warning。这是
FlagGems 注册 CUDA kernel 的预期提示，不是本例失败；最终以 `calls["n"]`
和输出检查为准。

调用路径是：

```text
torch.nn.functional.gelu
    -> FlagGems GELU

torch.ops.aten._scaled_dot_product_attention_math
    -> FlagOS-OP ops.sdpa_math A1
```

### 6.1 用计数器确认拦截

```python
calls = {"n": 0}
libs = register_a1(
    dispatch_key="AutogradCUDA",
    counter=calls,
    impl="p800",
)

with torch.no_grad():
    torch.ops.aten._scaled_dot_product_attention_math(
        q, k, v, None, 0.0, True, None, enable_gqa=False
    )

assert calls["n"] > 0
```

## 7. 典型使用场景

本节按使用目的组织场景。示例中的 `p800-kunlunxin`、`cambricon` 和
CPU 是当前已交付组合；其他平台接入时，以
[算子 / 平台矩阵](operator-catalog.md) 和算子内 `PLATFORM.md` 为准替换
profile、dispatch key 与实现入口。不要把某个平台 profile 当成固定接口
参数。

| 使用目的 | 场景 |
|---|---|
| 一条业务链路中混用 native、直调 kernel、FlagGems 与 A1 | 场景 A |
| FlagGems 负责周边标准算子，FlagOS-OP 负责 private 算子 | 场景 B |
| 同一进程注册多个 FlagOS-OP 算子 | 场景 C |
| 不修改进程级 dispatcher，只实验某个实现 | 场景 D |
| 在无目标硬件时验证语义或注册链路 | 场景 E |
| 使用基础 ATen 算子族做平台回归 | 场景 F |
| 比较 native baseline 与 A1 输出 | 场景 G |

### 场景 A：连续调用链，不同算子走不同映射

可运行样例：

```text
examples/operator-usage/example.py
```

链路：

```text
token ids
  -> embedding
  -> Q/K/V projection
  -> aten::_scaled_dot_product_attention_math
  -> GELU
  -> output projection
```

同一输入分别运行三种映射：

| 模式 | embedding | sdpa_math | GELU |
|---|---|---|---|
| native | native ATen | native private math | native ATen |
| direct | FlagOS-OP direct P800 kernel | FlagOS-OP direct P800 fast path | FlagGems direct API |
| a1-mixed | FlagOS-OP A1 | FlagOS-OP A1 | FlagGems dispatcher |

运行：

```bash
cd FlagOS-OP
PYTHONPATH="$PWD" python3 examples/operator-usage/example.py p800-kunlunxin
```

样例会输出每个算子的实际路由，并分别断言 embedding 与 `sdpa_math` 的
A1 counter 都等于 1。这展示的是框架接入常见形态：按算子语义和支持证据
选择映射，而不是把所有算子切换到同一个 backend。

### 场景 B：FlagGems 周边算子 + FlagOS-OP private 算子

适合条件：

1. 周边通用算子由 FlagGems 提供；
2. 目标 private 算子没有同语义 FlagGems 实现；
3. 需要保留 private op 的完整输出契约。

```python
import flag_gems
from ops.sdpa_math import register_a1

flag_gems.only_enable(include=["gelu"])
libs = register_a1("AutogradCUDA", impl="p800")
```

调用关系：

```text
F.gelu                                      -> FlagGems
aten::_scaled_dot_product_attention_math    -> FlagOS-OP
```

### 场景 C：多算子 A1 同时注册

多个不同 ATen 算子可以在同一进程注册，但必须分别持有注册对象：

```python
from ops.embedding import register_a1 as register_embedding
from ops.sdpa_math import register_a1 as register_sdpa_math

embedding_calls = {"n": 0}
attention_calls = {"n": 0}

embedding_lib = register_embedding(
    "AutogradCUDA",
    counter=embedding_calls,
    platform="p800-kunlunxin",
)
attention_libs = register_sdpa_math(
    "AutogradCUDA",
    counter=attention_calls,
    impl="p800",
)
```

要求：

1. 两个算子的 ATen 名称不同；
2. 保存 `embedding_lib` 和 `attention_libs`；
3. 分别检查 counter；
4. baseline 不在完成注册后的同一进程采集；需要对照时使用独立子进程。

### 场景 D：不想全局注册，直调 kernel

A1 会改变进程内 dispatcher 状态。一次性实验可以直调实现：

```python
from flag_gems import gelu
from ops.embedding.kernel import p800_kunlunxin as embedding_backend
from ops.sdpa_math.kernel.p800_fast_level import sdpa_math_p800_fast

embedded = embedding_backend.embedding(weight, ids)
context, probabilities = sdpa_math_p800_fast(
    query, key, value, None, 0.0, True, None, enable_gqa=False
)
activated = gelu(context)
```

适合实验、精确控制 kernel 参数或比较多种实现。需要旧业务代码零改动时，
仍应使用 A1。

### 场景 E：CPU / 语义参考

CPU 可作为语义锚点：

```python
import torch
from ops.sdpa_math import register_a1

libs = register_a1("CPU", impl="torch")
```

适合验证 operator contract、比较 golden，以及在无硬件环境下检查注册链路。

### 场景 F：Cambricon 基础 ATen 算子族

基础 ATen 算子族当前以 Cambricon MLU 为交付平台。具体算子列表以
[算子 / 平台矩阵](operator-catalog.md) 为准：

```python
from ops.clone import register_a1

calls = {"n": 0}
libs = register_a1(
    dispatch_key="AutogradPrivateUse1",
    counter=calls,
    platform="cambricon",
)
```

旧 profile / 单算子脚本模式可继续传 compatibility key：

```python
libs = register_a1("PrivateUse1", counter=calls)
```

### 场景 G：native baseline 与 A1 对照

正确方式是子进程隔离：

```text
child 1: 未注册，采集 native baseline
child 2: 注册 FlagOS-OP A1，采集注入结果
parent : 载入两边输出，比较精度
```

`examples/operator-usage/example.py` 已实现这个模式。不要在同一进程里
先注册 A1，再尝试用 `torch.ops` 采集未注册原生结果。

## 8. 注册边界

### 8.1 注册对象生命周期与进程隔离

`torch.library` 注册是进程级操作。当前实现要求保留返回的 `Library` 对象；
`del libs` 只删除 Python 引用，不能把它当作可靠的注销 API。不同 PyTorch
版本和注册路径的析构行为可能不同，因此基线、单元测试和 benchmark 都应
使用独立进程隔离。

因此：

- 原生 baseline 必须在注册前采集；
- 单元测试和 benchmark 建议子进程隔离；
- 长服务应在启动阶段确定算子策略。

### 8.2 避免同一个 ATen 算子双注册

如果 FlagGems 和 FlagOS-OP 都接管同一个标准 ATen 算子，注册顺序和最后
覆盖行为会造成混淆。推荐：

1. private / 无同语义 FlagGems 实现的算子直接注册 FlagOS-OP；
2. 标准 ATen 算子用 `flag_gems.only_enable()` 控制范围；
3. 不确定时用子进程分别跑 baseline 和 A1；
4. 不要在同一进程注册 A1 后再采集“未注册原生”结果。

### 8.3 平台 profile

| 算子 | P800 | Cambricon | CPU |
|---|---|---|---|
| `sdpa_math` | `p800-kunlunxin` | — | `cpu` |
| `sdpa` | `p800-kunlunxin` | `mlu590` | — |
| `embedding` | `p800-kunlunxin` | `cambricon` | — |
| 基础 ATen 算子族 | — | `cambricon` | — |

以 `ops/catalog.yaml` 和算子内 `PLATFORM.md` 为准。

## 9. 常见问题

### `ModuleNotFoundError: No module named 'ops'`

设置：

```bash
export PYTHONPATH="$FLAGOS_OP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

### `ModuleNotFoundError: No module named 'flag_gems'`

当前环境没有 FlagGems。优先使用团队提供的厂商运行时；开发者源码方式见
第 2 节。

### `ImportError: cannot import name 'register_a1'`

检查：

1. 是否从 `ops.<operator>` 导入；
2. 仓库根目录是否在 `PYTHONPATH`；
3. 检出提交是否包含该算子的包化入口。

正确示例：

```python
from ops.sdpa_math import register_a1
```

### 注册了但没有命中

检查：

1. 是否保存 `register_a1()` 返回的 lib 引用；
2. dispatch key 是否匹配平台；
3. 是否用 `counter` 确认调用进入 A1。

## 10. 最小 checklist

```text
[ ] git clone FlagOS-OP
[ ] export FLAGOS_OP_ROOT 和 PYTHONPATH
[ ] import torch / flag_gems 成功
[ ] import ops.<operator> 成功
[ ] check_env 通过或记录差异
[ ] 先跑 example.py
[ ] 应用中使用 only_enable 控制 FlagGems 范围
[ ] register_a1 后保存 lib 引用
[ ] 用 counter 验证 A1 命中
[ ] baseline 与 A1 用子进程隔离
```
