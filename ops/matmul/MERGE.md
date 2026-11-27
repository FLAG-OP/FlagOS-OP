# 合并流程（MERGE）— ops/matmul 公共骨架

> 本目录为公共算子骨架, thead-ppu 首交付（issue #55）; PR #35（昆仑芯
> P800）与 PR #53（寒武纪 MLU590）在途。合入按以下七步执行, 平台范围
> 不互相推广（issue #55 明确排除项条款）。

1. **基线对齐**: 基于 main 最新 `ops/matmul`（本骨架）重建 PR 分支。
2. **平台文件合并**: `kernel/` 下按 sdpa 范式启用 `kernel/backends/`
   选择器（第二平台出现时）; 各平台 backend 保留自己的守卫与绑定文档。
3. **register 合并**: `_backend_for` 扩表 + `_probe_ok` 各平台守卫
   （xmlir → p800、torch_mlu → mlu590、PPU 栈 → thead）; key 映射按
   各自 PLATFORM.md（P800=AutogradCUDA、MLU=AutogradPrivateUse1、
   PPU=AutogradCUDA）。
4. **catalog 合并**: `ops/catalog.yaml` 的 platforms 按 family 合并,
   逐 profile 证据记录不共享（issue #41 口径）; 跑
   `python3 scripts/build_ops_index.py --check`。
5. **perf 注册合并**: `common/perf_registry.py` 的 provider 名不冲突
   （ops.matmul.<impl> 命名空间）; 形状组按平台分档。
6. **测试合并**: 三层测试的平台分段用 `platform=` 显式守卫;
   `--all` 同进程串跑不残留（钉选/env 恢复惯例）。
7. **回归验收**: 各平台 `python3 example.py <profile>` 三层全绿 +
   `build_ops_index --check` + `evidence_check` + `pytest tests/unit`。
