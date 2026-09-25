# embedding 测试报告

| 项 | 值 |
|---|---|
| 日期 | 2026-09-23（cambricon 平台接入与复测 2026-09-24） |
| 设备 | p800-kunlunxin `cuda:1` · cambricon MLU590 `mlu:0` |
| 环境 | P800: Torch 2.9.0+cu129 / torch_xmlir / Triton 3.0；MLU: Torch 2.11.0+cpu / torch_mlu 1.33.1 / Triton 3.4.0+mlu2.1.1 |
| 结论 | 通过（两平台） |

## 1. 测试范围

| 层级 | 覆盖 | 结果 |
|---|---|---|
| kernel forward | 1D/2D/3D、D8/80/512、empty、int32、padding | ✅ |
| kernel backward | FP32/FP16/BF16 × dense/scale_freq × duplicate/padding | ✅ |
| 黄金 | 174 组 | ✅ |
| A1 | 拦截、direct bitwise、autograd | ✅ |
| 应用层 | `nn.Embedding` + MLP forward/backward/greedy | ✅ |
| FlagOS 栈 | `flag_gems.only_enable(['gelu'])` 后运行应用层 | ✅ |
| 边界 | sparse forward 接受、sparse backward / invalid padding 拒绝 | ✅ |
| 第二平台 cambricon | kernel / op / framework / guard 同判据复跑 | ✅ |

## 2. 环境

| 项 | 值 |
|---|---|
| Python | 3.10.18 |
| Torch | 2.9.0+cu129 |
| torch_xmlir | XMLIR--bc1b1dc6f-dev+2026032411 |
| Triton | 3.0.0+03c4c9be |
| 设备 / 卡 | p800-kunlunxin / `cuda:1` |

## 3. 精度结果

复现命令：

```bash
python3 example.py p800-kunlunxin
python3 script/gen_golden.py
python3 script/check_accuracy.py --impl p800 --device cuda:1
python3 script/check_accuracy.py --impl triton --device cuda:1
python3 script/check_accuracy.py --impl native --device cuda:1
pytest -q tests/unit

# 第二平台 cambricon（MLU）
MLU_VISIBLE_DEVICES=7 python3 example.py cambricon
MLU_VISIBLE_DEVICES=7 python3 script/check_accuracy.py --impl cambricon --device mlu:0
```

结果摘要：

- kernel forward: 20/20，max error 0；
- kernel backward: 6/6，max error 0；
- golden: 174/174，max error 0；
- A1: interception count 5，registered path equals direct path bitwise；
- dense grad error 0；
- scale-grad error 0；
- application logits/gradient diff 0；
- greedy continuation match 1.00。
- sparse=True forward 与 dense forward bitwise equal；
- sparse backward显式拒绝。

## 4. 性能结果

```bash
python3 script/bench_perf.py --device cuda:1 --dtype float16 \
  --json-out reports/perf_fp16_p800-kunlunxin.json
python3 scripts/perf_run.py --device p800-kunlunxin --pattern ops.embedding
python3 scripts/perf_compare.py --device p800-kunlunxin
```

perf gate 结果：`FAIL 0 · WARN 0 · NEW 0`。入库基线中
`ops.embedding.p800.forward` 为 0.051ms，native 对照为 0.052ms。
A1 dispatch 独立子进程测量（P800）：native 0.0489ms，A1 0.0730ms，
附加约 0.0241ms——**该 a1 数字已作废**：当时 `bench_dispatch.py` 裸调
`register_a1`，返回的 `Library` 被 GC 即反注册，实测的是未注册状态
（修复与说明见 [cambricon.md](cambricon.md) §7）；native/direct 行与
精度结论不受影响。P800 需在有 `torch_xmlir` 的环境复跑。

## 5. 问题与风险

| 风险 | 状态 |
|---|---|
| `scale_grad_by_freq=True` fallback 有额外中间 tensor | 已量化：约 1.96ms vs dense 1.33ms（MLU 走原生，+11%） |
| Triton gather 慢 | 仅保留探针，不接生产（MLU 131k 还撞 grid 65535 上限） |
| sparse backward 未覆盖 | 显式拒绝，符合需求边界 |
| MLU `sparse=True` 反向返回 COO 而非报错 | `kernel/cambricon.py` 显式 `NotImplementedError`，守卫用例覆盖 |
| P800 A1 dispatch 数字作废（Library GC） | 脚本已修；MLU 复测 +0.059ms，P800 待有 `torch_xmlir` 环境复跑 |
| `check_env --device cambricon` 6 项不符 | 既有锁漂移（`d5c7a3d` 写锁的容器镜像 ≠ 本机 torch2.11 栈），与本改动无关；`--device mlu590` 为 OK |

## 6. cambricon（MLU590）复跑结果

```bash
MLU_VISIBLE_DEVICES=7 python3 test/kernel_level.py cambricon
MLU_VISIBLE_DEVICES=7 python3 test/op_level.py cambricon
MLU_VISIBLE_DEVICES=7 python3 test/framework_level.py cambricon
MLU_VISIBLE_DEVICES=7 python3 example.py cambricon
python3 -m pytest tests/unit -q
```

| 层级 | 结果 |
|---|---|
| kernel forward | ✅ 20/20，`max_forward_err=0.0` |
| kernel backward | ✅ 6/6，`max_backward_err=0.0`（3 dtype × scale 0/1） |
| 守卫 | ✅ sparse backward 拒绝 / invalid padding 拒绝 / sparse forward 接受 |
| 黄金 | ✅ 174/174，worst=0.000e+00（`--impl cambricon`，逐位判据） |
| A1 op | ✅ 拦截 5 次；注册=直调（逐位）；dense/scale grad err=0；sparse fwd OK / bwd rejected |
| 应用层 | ✅ 拦截 8 次；logits diff=0.000e+00；greedy 1.00；embedding/首层 linear 梯度 diff=0；`flag_gems.only_enable(['gelu'])` 生效 |
| 一键三层 | ✅ `example.py cambricon` 1/3 kernel · 2/3 A1 op · 3/3 application 全 PASS |
| 单测 | ✅ `pytest tests/unit -q` 20 passed |
| perf gate | ✅ `FAIL 0 · WARN 0 · NEW 0`（`ops.embedding.cambricon.forward` 0.060ms） |

## 7. 结论

kernel 层、op 层、应用层、黄金与性能门禁在 **p800-kunlunxin 与 cambricon
两个平台上全部通过**；建议合入。
