# aten.scatter_ — Triton / A1（Ascend910）

无归约原地索引写入。已在 **Ascend910**（torch_npu / triton-ascend / PlatformFL）
完成 kernel、A1、黄金、framework 与**真实 vLLM C2**（src 生成阶段命中 8 次，
输出与基线逐 token 一致）；value 为 C1。参考 P800 gems5 的 PR #62：
其 kernelGen TLE/DSA 初稿在本栈编译失败，本实现为**纯 Triton**。

## 接口与范围

`scatter_triton(self, dim, index, src)` 返回同一个 `self`。
覆盖 `aten.scatter_.src` / `aten.scatter_.value`；不接管 reduce 重载。
支持任意 rank、负 dim、非连续 stride、storage offset、空 index、标量张量。
源张量与目标 dtype 一致，非空 index 为 int64；不广播 src/index。
重复索引的 tensor-src 写入胜者不确定，与原生接口相同。
目标内部重叠、输入别名、complex/量化 dtype 和直接调用的 autograd
不在支持范围，明确报错。float16/bfloat16/float32/float64、
int8/uint8/int16/int32/int64、bool 已有设备用例。

## 运行

```bash
# CPU only
python3 -m pytest tests/unit/test_scatter_metadata.py -q
python3 ops/scatter_/script/gen_golden.py
python3 ops/scatter_/script/check_accuracy.py --impl torch --device cpu
# 取到空闲 NPU、配置好可见卡后
python3 ops/scatter_/example.py ascend910
python3 ops/scatter_/script/check_accuracy.py --impl triton --device ascend910 --torch-device npu:0
# 真实 vLLM（C2）
python3 ops/scatter_/test/vllm_runner.py --mode prepare  --model /tmp/scatter-qwen3-tiny
python3 ops/scatter_/test/vllm_runner.py --mode baseline --model /tmp/scatter-qwen3-tiny --out /tmp/base.json
python3 ops/scatter_/test/vllm_runner.py --mode injected --model /tmp/scatter-qwen3-tiny --out /tmp/inj.json
python3 ops/scatter_/script/check_vllm.py /tmp/base.json /tmp/inj.json --out /tmp/c2.json
```

## 平台绑定（Ascend910）与内核设计

- A1：注册 `scatter_.src` / `scatter_.value` 到 **AutogradPrivateUse1**（+PrivateUse1）；
  `_version` 由实现递增（Autograd key 下 dispatcher 不自增）。
- 纯 Triton：host 端按 `index.shape` 计算 `base_off`（dst 非 dim 维）与
  `src_base`（全维）偏移，kernel 内 `dst_off = base + idx*stride_dim`，
  `tl.store` 写回；地址 int64，**BLOCK=1024**（4096 触发 UB 溢出）。
- 越界在写入前由 Python 侧 `any()` 检查并抛错（该栈 device_assert 不可靠）。
- fp64 回退原生 redispatch（BiSheng 不支持 64-bit float）。
- 无 autotune；启动置于 `torch_device_fn.device` 上下文。
- 平台细节见 [PLATFORM.md](PLATFORM.md)，证据见
  [reports/vllm.md](reports/vllm.md)（C2）与 [reports/ascend910.md](reports/ascend910.md)。
  P800 参考材料归档于 [reports/p800-reference/](reports/p800-reference/)。

## 合并前检查

- [x] 核心八件及包化入口
- [x] 设备三层测试（kernel 41+11、op 41、framework 1）与黄金 41/41
- [x] 真实 vLLM C2（src，top-k/top-p 路径，输出一致、命中 8）
- [x] 登记 catalog ascend910 C2 证据（complete-pending-review）
- [x] `pytest tests/unit` 96 passed；包化扫描通过

参考 Dashener2 的 PR #62（P800）与本仓库 operator 模板。
