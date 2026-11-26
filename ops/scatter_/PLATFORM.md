# 平台绑定清单（Platform Binding）— scatter_

> 本轮交付：**ascend910**（普通 Triton）。参考 P800 gems5 的 PR #62：
> 其 kernelGen 生成的 TLE/DSA 版本在本栈 `TritonToLinalgIncubated` 编译失败，
> 故改写为**纯 `tl.load`/`tl.store` + 计算偏移**的普通 Triton 实现（语义一致）。

## 1. 可直接复用（平台无关）

| 内容 | 位置 |
|---|---|
| 原生参考（判卷标准） | `reference.py`（`self.scatter_(dim,index,src)`） |
| 元数据/语义校验 | `validation.py` |
| 用例与边界 | `test/cases.py`、`test/edge_cases.py` |
| 黄金与脚本 | `goldendata/`、`script/` |

## 2. ascend910 绑定

| 项 | 结论 |
|---|---|
| A1 拦截 key | **AutogradPrivateUse1**（成对兼容 PrivateUse1；torch_npu 的 Autograd 包装不 redispatch） |
| 注册重载 | `scatter_.src` 与 `scatter_.value`（同一 `apply/value`） |
| 实现 | 普通 Triton：host 端按 `index.shape` 算 `base_off`（dst 非 dim 维）与 `src_base`（全维）偏移，kernel 内 `dst_off = base_off + idx*stride_dim`，`tl.store` 写回 |
| 设备上下文 | `flag_gems.runtime.torch_device_fn.device` |
| UB tile | **BLOCK=1024**（int64 索引/偏移，尤其 4096 会触发 `ub overflow` 192KB） |
| 越界检查 | Python 侧 `((index<0)|(index>=limit)).any()`，写入前抛错（`tl.device_assert` 不可靠） |
| dtype | fp16/bf16/fp32/int8/uint8/int16/int32/int64/bool；**fp64 回退原生 redispatch**（BiSheng 不支持 64-bit float） |
| 原地语义 | 返回同一张量；`_version` 由实现 `increment_version` 递增（注册 Autograd key 后 dispatcher 不自增） |
| 证据 | kernel 41 + 边界 11；op 41；黄金 41/41；framework 1；真实 vLLM C2 命中 8、输出一致 |
| 级别 | **C2**（src，真实 vLLM top-k/top-p 命中，complete-pending-review）；value 为 C1 |

## 3. 参考（P800 / PR #62）

P800 版本用 `triton.experimental.tle`（DSA/UB）实现；本栈 TLE 的串行 `dsa.copy`
在 `TritonToLinalgIncubated` 阶段报错（40/41 失败）。普通 Triton 版本在本栈 41/41 通过。

## 4. 复现

```bash
python3 ops/scatter_/example.py ascend910
python3 ops/scatter_/script/gen_golden.py
python3 ops/scatter_/script/check_accuracy.py --impl triton --device ascend910 --torch-device npu:0
```
