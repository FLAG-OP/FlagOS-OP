# 算子总体报告: _softmax（aten::_softmax）

> 交付批次: issue #55（平头哥真武 810E, P0/P1-1124 台账）· 2026-11-25/26
> 设备: PPU-ZW810E（真武 810E, 卡 6, ppu-smi 空闲确认） · 环境: flagos-ppu-wt-dev 基座: torch 2.10.0 / triton 3.6.0 / flagtree 0.6.1+ppu3.6（wheel 损坏在册）/ python 3.12.3; 无 flag_gems
> 本页是验证矩阵汇总; 使用入口见 README, 平台绑定见 PLATFORM, 实验数据在 reports/。

## 实现矩阵

| 平台 | 级别 | 文件 | 生产路径 | 状态 |
|---|---|---|---|---|
| thead-ppu | A1 · native 委托 | kernel/thead_ppu.py + kernel/torch_level.py | 原生委托（原语 + 解析反向; h2f 仅 Half 边界回退） | ✅ |

## 验证矩阵（2026-11-25/26, 真机 卡 6）

| 层级 | 入口 | 结果 |
|---|---|---|
| kernel 层 | `test/kernel_level.py thead-ppu` | ✅ |
| op 层 | `test/op_level.py thead-ppu` | ✅ |
| 应用层 | `test/framework_level.py thead-ppu` | ✅ |
| 黄金 | `script/check_accuracy.py` | native: _softmax: native @ cuda:6 24 / 24 PASS, worst=6.104e-05 |
| 性能 | `script/bench_perf.py` | 见 [reports/performance.md](reports/performance.md) |

## 证据等级

- 当前: **C1**（framework 层 mini consumer 基线/注入双跑, 恒等/梯度对拍;
  mini consumer 只能支撑 C1——docs/testing.md 证据等级表）。
- C2（真实 vLLM EngineCore 双跑）: 归 thead-ppu-c2 栈（flagrelease_ppu_vllm024_gems_540,
  flag_gems 5.4.0 + 插件 0.3.0rc2.post3 + vllm 0.24.0）执行, 待排期;
  按验收标准 C2 合并门禁清单执行, 完成前不写「C2 成立」。

## 遗留与边界

- inference_mode 下注册零命中（known-issues #19.3, Autograd* 键通病）——
  如实登记为平台边界, 不影响 grad/no_grad 口径拦截。
- 平台分化与绑定结论逐条见 [PLATFORM.md](PLATFORM.md)。
