# 测试体系

## 两级测试

| 层级 | 验证什么 | 形式 |
|---|---|---|
| 算子层 | kernel 本身: 精度 / dispatch 选择 / 性能 | 三段式，秒级 |
| 框架层 | 算子在真实 vLLM 前向被调用且不破坏输出 | 基线/插件双跑比对，分钟级 |

### 算子层三段式

1. **精度**: 实现 vs PyTorch 参考，多 shape×dtype；
   容差 fp32=1e-5, bf16/fp16=1e-2（先 .float() 再比）
2. **dispatch**: 注册数 / `with_preference` 策略切换 / 正确选中
3. **性能**: warmup + 短采样（100 次内）+ synchronize；
   长循环会触发输出分配器池增长，均值失真一两个数量级

### 框架层断言策略（两种）

| 场景 | 断言 | 原因 |
|---|---|---|
| 数值恒等路径（aten 恒等覆盖 / audit reference 委托） | 前 N token 完全一致 | 数学上恒等，位级可比 |
| 自定义数值实现（Triton） | 前 2 token 一致率 ≥ 2/3×N | 随机权重+贪心解码把数值微差混沌放大为后期分叉 |

恒等断言的前缀长度 N **自适应**: min(上限 8, 设备黄金实测最短稳定前缀)。

## 测试输入模板

- 声明式定义: `inputs/spec.yaml`（fixed/length_sweep/edge_cases 文本类
  + token_based 算子类），`scripts/gen_inputs.py` 生成产物
- 同一 spec 在任何硬件生成完全相同的输入——黄金可比的前提
- 框架级运行统一 `tokenize + 词表钳制` 输入（`PROMPTS_AS_TOKENS=1`），
  规避 vendor 栈 embedding 静默越界（known-issues #8）

## 黄金输出（golden outputs）

```bash
python3 scripts/build_golden.py --device cpu                 # transformers CPU 权威参考
python3 scripts/build_golden.py --device p800-kunlunxin      # vLLM reference 路径 ×3 快照
python3 scripts/compare_golden.py --devices p800-kunlunxin,cpu
```

- 黄金 = N 次独立运行快照集合 + 逐位置多数票共识前缀
- 落盘 `golden/<device>_golden.json`（含引擎指纹/prompts 哈希，应入库）
- 用途: 跨会话回归检测（FlagOS/torch/厂商栈升级改变数值行为时报警）
- CPU 设备走 HuggingFace transformers 直接推理——最权威语义参考，
  完全绕过 vLLM/厂商栈，确定性

## 漂移实验

```bash
python3 scripts/drift_study.py --device p800-kunlunxin --runs 8 --mode reference
python3 scripts/drift_study.py --device p800-kunlunxin --runs 4 --mode vendor
```

量化跨进程非确定性（逐位置一致率 / 两两首分歧 / 全员一致前缀），
用于校准断言前缀。本机结论: 受控条件下 reference 8 连跑 + vendor
4 连跑全部 100% 确定；历史偶发后期 token 漂移为低概率条件相关事件。
