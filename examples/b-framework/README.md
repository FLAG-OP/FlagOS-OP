# 样例: B 路线 × 框架层

## 目标

验证 vendor backend（audit 计数+委托模式）在**真实 vLLM 前向**
中被选中执行，纯拦截不改变输出，且黄金回归通过。

## 机制

- audit vendor（计数 + reference 委托）经 `VLLM_FL_PLUGIN_MODULES` 注入
- 基线跑与插件跑都钉 `PER_OP=silu_and_mul=reference`
  （唯一差异 = 是否经 audit 拦截 → 数值恒等可严格断言）
- 叠加黄金共识回归（跨会话漂移检测）

## 为什么委托 reference 而非厂商 kernel

实测本机 `xtorch_ops.swiglu` 独立调用不写输出（已知问题#5），
厂商委托无法保证数值恒等；profile 无厂商委托时 audit 自动
退化为 reference 委托。

## 运行（约 2 分钟）

```bash
python3 examples/b-framework/example.py
```

## 预期输出

```
audit silu calls   : >0
output compare     : first N tokens identical
golden check       : matches golden consensus (prefix N)
```
