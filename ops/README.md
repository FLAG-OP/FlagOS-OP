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
| [sdpa](sdpa/)（scaled_dot_product_attention） | Triton / 厂商委托 | A1（aten 拦截） | ✅ 定稿 | [REPORT](sdpa/REPORT.md) | ascend910 + p800-kunlunxin + mlu590 · 三平台黄金 397/397（Ascend 历史 265/265） · [平台绑定](sdpa/PLATFORM.md) / [多平台合并](sdpa/MERGE.md) |

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
