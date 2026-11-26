# aten.scatter_ — Triton / A1

无归约原地索引写入，已在昆仑芯 P800 XMLIR gems5 栈完成三层验证。
src 重载已补齐真实 vLLM C2（随机权重 Qwen3）；value 为 C1 mini consumer。
性能目前低于原生，适合作为兼容性开发样例。

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
# 取得空闲设备使用许可、配置好可见卡后才执行
python3 ops/scatter_/example.py p800-kunlunxin-gems5 --torch-device cuda:0
python3 ops/scatter_/script/check_accuracy.py --impl triton --device p800-kunlunxin-gems5 --torch-device cuda:0
python3 ops/scatter_/script/bench_perf.py --device p800-kunlunxin-gems5 --torch-device cuda:0 --out scatter-perf.json
```

预期：三层分别返回 `ok: True`，黄金全部精确相等；性能无预设加速承诺。
默认 example 应用层为 token-selection mini consumer（C1）。
真实 vLLM 单独运行，复现及证据范围见 [reports/vllm.md](reports/vllm.md)。
注册仅覆盖 CUDA backend，让原生 ADInplaceOrView 维护版本计数，
Library 必须保持引用；测试结束销毁注册。算子包导入不加载 Triton。

## 内核设计

一个线程处理一个 index 元素，以 index shape 解码坐标，分别计算
index/src/dst 的 stride 地址。地址使用 int64；整块 block=256，
元素数整除 16384 时用大块，尾部使用独立 BLOCK=1 程序。
XMLIR 的 masked scatter 和重复尾写实测会产生错值，不能依靠掩码补齐。
该栈忽略 device_assert，因此在写入前执行索引范围检查并同步判定；
这是额外性能开销。无 autotune；启动置于 `torch_device_fn.device` 上下文。

## 合并前检查

- [x] 核心八件及包化入口
- [x] 设备三层测试、黄金验证、性能实测
- [x] 登记 catalog 平台证据
- [x] CI 等价全库包化扫描 17/17、基础设施回归 186/186
- [x] 可复现环境和测试结果写入报告

仓库 `ops/test_packaging.py` 还包含 Ascend 专用端到端测试，P800 环境
不执行该部分；这里的包化结果是 CI 的逐目录导入/导出扫描。

参考 Dashener2 的 PR #5、#34、#42 及仓库 operator 模板。
