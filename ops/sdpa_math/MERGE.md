# 多平台合并流程（MERGE）— sdpa_math

> 新平台实现合入本目录的固定流程（裁剪自 ops/sdpa/MERGE.md 与
> ops/embedding/MERGE.md 范式）。核心原则：**黄金/测试/注册链复用，
> 只写 kernel 本体**。

## 七步流程

1. **黄金复用**: `goldendata/` 平台无关（CPU 生成），新平台直接跑
   `script/check_accuracy.py`——不过黄金不谈合并
2. **kernel 本体**: `kernel/<platform>_level.py`（现有：
   `triton_level.py` ascend910 路线 / `p800_fast_level.py` vendor 路线）
3. **register 平台分发**: `register.py` 的 `_fwd`/backend 选择加平台
   分支（含 `_DEVICE_PROBE` 能力守卫参考 ops/embedding）
4. **_profile 探测链**: 平台 try-import 段（对齐 sdpa/embedding 顺序）
5. **三层验证**: `python3 test/{kernel,op,framework}_level.py <platform>`
   + `example.py <platform>` 一键
6. **包化回归**: `python3 ops/test_packaging.py`（全算子导入扫描）
7. **文档同步**: README/REPORT 平台列、reports/ 增量、PLATFORM.md
   绑定项、本文件、CHANGELOG

## 合并冲突解决范式（库级惯例）

- 文档冲突: 双方内容**并列**（验证矩阵按平台列拆行），不删任何一方
- 代码冲突: 统一采 platform facade / backend 选择器新范式
- rebase 后必须: 本机全链回归 + `ops/test_packaging.py` 双绿才可合并

## 绑定速查

见 [PLATFORM.md](PLATFORM.md)（ascend910 64×64 tile 上限 / p800
vendor O/LSE + exact-P）。
