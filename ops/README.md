# 已开发算子（ops/）

这个目录存放**按样板开发完成并交付的算子**——每个子目录一个算子，
结构与 [templates/operator](../templates/operator/README.md) 完全一致
（kernel 三级 · test 三层 · goldendata · script · REPORT + 交付报告四件套）。

## 新算子放进来

```bash
cp -r templates/operator ops/<你的算子名>/
cd ops/<你的算子名>/
grep -rl my_op . | xargs sed -i 's/my_op/<你的算子名>/g'
```

开发流程照[全链路指南](../docs/fullstack-guide.md)七步走；完成后:

1. `python3 example.py <profile>` 一键三层全绿
2. 补齐交付报告四件套（REPORT + development + test-report + 分册，
   范本见 [softmax-fullstack](../examples/softmax-fullstack/reports/development.md)）
3. 在下方索引表登记一行
4. 交付路径选择见[集成指南](../docs/integration.md)（FlagGems 贡献 /
   aten 注册 / vllm_fl 插件 / 厂商身份）

## 算子索引

| 算子 | 开发级别 | 路线 | 状态 | 交付报告 | 备注 |
|---|---|---|---|---|---|
| [type_as](type_as/) | Triton 级 | A1 aten | 三层全绿 | [REPORT.md](type_as/REPORT.md) | 黄金 111/111；位级一致 |
| [clone](clone/) | Triton 级 | A1 aten | 三层全绿 | [REPORT.md](clone/REPORT.md) | 黄金 48/48；存储独立 |
| [contiguous](contiguous/) | Triton 级 | A1 aten | 三层全绿 | [REPORT.md](contiguous/REPORT.md) | 黄金 114/114；转置 swiz 提速 ~80x |
| [copy_](copy_/) | Triton 级 | A1 aten | 三层全绿 | [REPORT.md](copy_/REPORT.md) | 黄金 78/78；原地+广播 |
| [dropout](dropout/) | Triton 级 | A1 aten | 三层全绿 | [REPORT.md](dropout/REPORT.md) | 黄金 189/189；随机分支统计判定 |
| [empty_like](empty_like/) | torch 级 | A1 aten | 三层全绿 | [REPORT.md](empty_like/REPORT.md) | 分配/元数据；黄金 168/168（元数据） |
| [empty](empty/) | torch 级 | 自用/实验 | 三层全绿 | [REPORT.md](empty/REPORT.md) | 分配原语，不注册（递归风险）；元数据 48/48 |
| [empty_strided](empty_strided/) | torch 级 | A1 aten | 三层全绿 | [REPORT.md](empty_strided/REPORT.md) | 分配/元数据；元数据 30/30 |
| [detach](detach/) | torch 级 | A1 aten | 三层全绿 | [REPORT.md](detach/REPORT.md) | 别名词义；黄金 36/36 |
| [detach_](detach_/) | torch 级 | 自用/实验 | 三层全绿 | [REPORT.md](detach_/REPORT.md) | 不可 A1 注册（autograd key）；36/36 |
| [item](item/) | torch 级 | A1 aten | 三层全绿 | [REPORT.md](item/REPORT.md) | host 标量；黄金 36/36 |
| [_local_scalar_dense](_local_scalar_dense/) | torch 级 | A1 aten | 三层全绿 | [REPORT.md](_local_scalar_dense/REPORT.md) | host 标量原语；36/36 |
| [result_type](result_type/) | torch 级 | 自用/实验 | 三层全绿 | [REPORT.md](result_type/REPORT.md) | dtype 元数据；黄金 8/8，矩阵 64 项 |

> 均为 MLU（cambricon）实现。合计 13 个算子。
> 应用层：各 `test/framework_level.py` 为快速等价验证；另有**真实 vLLM** 的
> A1 多算子注入验证见 `tests/framework_level/test_a1_aten_ops.py`
> （sitecustomize 桥 + 基线/注入双跑，断言输出逐位一致）。
> 注：`detach` 可单独注册，但在真实 vLLM 建模时会破坏 `nn.Parameter`
> 构造，故不纳入多算子注入集（与 `detach_` 同属进程内不可共存）。

> 结构同构的教学样例 [softmax-fullstack](../examples/softmax-fullstack/)
> 保留在 examples/（它的定位是"教"，ops/ 的定位是"用"）；
> 后续正式开发的算子一律进 ops/ 并在此登记。

## 约定

- **目录名 = 算子名**（小写下划线，如 `gelu_and_mul`）
- 未实现的级别**置空 + 说明**（`kernel/hardware_level/README.md`），
  不留空文件
- 命名/级别/路线术语遵循 [体系结构](../docs/architecture.md)；
  Triton kernel 必须内置 [#11](../docs/known-issues.md) device 上下文与
  [#15](../docs/known-issues.md) 尾块防护两条硬约束
- 合入本目录前: 三层测试全绿 + 黄金回归 + perf 门禁通过
  （见[集成指南交付 checklist](../docs/integration.md#checklist)）
