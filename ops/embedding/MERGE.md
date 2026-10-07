# 多平台合并流程（MERGE）— embedding

> 新平台实现合入本目录的固定流程。范式实证：ascend910（PR#11）→
> cambricon（PR#9 rebase 合入）。核心原则：**黄金/测试/注册链复用，
> 只写 kernel 本体**。

## 七步流程

1. **黄金复用**：`goldendata/` 平台无关（CPU 生成），新平台直接跑
   `script/check_accuracy.py`——不过黄金不谈合并
2. **kernel 本体**：`kernel/<platform>.py` 四件套——`embedding` /
   `embedding_backward` / `PLATFORM` / `_DEVICE_PROBE`（注册守卫
   能力探测）
3. **facade 注册**：`kernel/platform.py` 三处——`_BACKENDS` 设备
   映射、`DISPATCH_KEYS`、`synchronize` 分支
4. **register 选择器**：`register.py` 的 `_BACKENDS` dict 加一行；
   dispatch_key 冲突时（如 npu/mlu 同为 AutogradPrivateUse1）依赖
   `platform=` 显式参数
5. **_profile 探测链**：`_profile.py` 加 try-import 段（顺序与 sdpa
   对齐：mlu → xmlir → npu → cpu）
6. **三层验证**：`EMBEDDING_PROFILE=<platform>` 跑 kernel/op/framework
   三层 + `example.py` 一键
7. **文档同步**：README/REPORT 平台列、reports/ 增量、本文件与
   PLATFORM.md 绑定项、CHANGELOG

## 合并冲突解决范式（PR#9 rebase 实证）

- 文档冲突：双方内容**并列**（验证矩阵按平台列拆行），不删任何一方
- 代码冲突：统一采 **platform facade** 新范式（旧 `_load_backend` /
  裸 `from kernel import` 全部迁移）
- rebase 后必须：本机全链回归 + `ops/test_packaging.py`（多算子
  同进程）双绿才可合并

## 验证矩阵终态（三平台）

| 层 | 命令 | 结果 |
|---|---|---|
| kernel（ascend910） | `EMBEDDING_PROFILE=ascend910 python3 test/kernel_level.py` | 20 前向+6 反向 err=0.0 |
| op（ascend910） | `test/op_level.py` | 拦截 5 次·逐位·梯度 err=0.0 |
| kernel/op/应用（p800） | 同上 | 20/20+6/6 err=0；A1 拦截 5 次 |
| kernel/op/应用（cambricon） | `EMBEDDING_PROFILE=cambricon ...` | 见 reports/cambricon.md |
| 包化共存 | `python3 ops/test_packaging.py` | 16 算子导入+双注册+消费 全绿 |
