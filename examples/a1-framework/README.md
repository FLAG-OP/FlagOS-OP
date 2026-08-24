# 样例: A1 路线 × 框架层

## 目标

验证 aten 路线的自定义算子（数值恒等的 aten::silu 计数覆盖）
在**真实 vLLM 前向**中被调用，且不改变端到端输出。

## 框架级三要素（本样例的核心知识）

| 要素 | 机制 | 为什么需要 |
|---|---|---|
| sitecustomize 注入 | `injection/sitecustomize.py` 加入 PYTHONPATH，子进程启动即注册 | vLLM v1 前向在 EngineCore 子进程，主进程 torch.library 注册不传播 |
| FlagGems 黑名单 | `VLLM_FL_FLAGOS_BLACKLIST=silu,silu_` | flag_gems.enable() 也会注册 silu，防止覆盖计数实现 |
| PER_OP 钉路径 | `VLLM_FL_PER_OP=silu_and_mul=reference` | 迫使模型走 reference → F.silu → 命中 aten::silu |

## 断言策略

1. 调用计数 > 0（真实前向命中）
2. 与基线前缀恒等——前缀长度**自适应**于设备黄金实测稳定性
3. 黄金共识回归（跨会话漂移检测）

## 运行（约 2 分钟，含 2 次真实 vLLM 推理）

```bash
python3 examples/a1-framework/example.py
```

## 预期输出

```
aten::silu calls   : >0
output compare     : first N tokens identical [exact-prefix-N]
golden check       : matches golden consensus (prefix N)
```
