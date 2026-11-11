# 多平台合并流程（MERGE）— matmul

本流程用于把 matmul 从昆仑芯 P800 扩展到寒武纪 MLU590。黄金和测试
逻辑复用，平台结论必须重新采集。

## 1. 锁定环境

确认 `torch_mlu`、Triton、FlagGems 版本与
`configs/env/mlu590.lock.yaml` 一致：

```bash
python3 scripts/check_env.py --device mlu590
```

## 2. 确认可见设备

确认 `MLU_VISIBLE_DEVICES`、逻辑设备 `mlu:0` 和共享卡占用情况。
测试前记录卡号和环境变量，不在报告里写未经验证的物理映射。

## 3. 复用 CPU 黄金

黄金只由 CPU FP32 参考生成：

```bash
python3 ops/matmul/script/gen_golden.py
```

不要使用昆仑芯设备输出重建寒武纪黄金。

## 4. 先跑最小 kernel

先测单个矩阵乘和单侧反向，再跑完整 144 组。最小阶段必须确认：

```text
mlu:0 可用
tl.dot 可编译
前向有限值
dA/dB 形状正确
重复运行结果一致
```

## 5. 验证 A1 注册

使用 `AutogradPrivateUse1` 注册 `aten::matmul`，分别验证：

- `torch.matmul`；
- `torch.ops.aten.matmul.default`；
- CPU 调用不被拦截；
- 前向计数增加；
- 左右输入一阶梯度。

## 6. 跑三层测试

```bash
python3 ops/matmul/example.py mlu590
```

若需要定位失败，再分别运行 `test/kernel_level.py`、
`test/op_level.py` 和 `test/framework_level.py`。

## 7. 采集寒武纪独立性能

```bash
python3 scripts/perf_run.py --device mlu590 --pattern ops.matmul
python3 scripts/perf_compare.py --device mlu590
```

寒武纪的基线必须单独建立。不得把 P800 的耗时、门禁或加速比复制到
寒武纪报告。

## 8. 更新证据和目录

只有实机测试通过后，才允许：

1. 将 `ops/matmul/reports/cambricon.md` 更新为真实结果；
2. 在 `ops/catalog.yaml` 的 matmul `platforms` 中加入 cambricon；
3. 更新 `ops/README.md`；
4. 在 `REPORT.md` 增加寒武纪证据入口；
5. 保留原昆仑芯结论，不覆盖、不改写。

## 9. 合并门禁

```bash
python3 scripts/build_ops_index.py --check
python3 scripts/intake_validate.py --validate-only
```

最终报告必须如实区分 P800 已完成、寒武纪本轮完成和尚未验证项。

