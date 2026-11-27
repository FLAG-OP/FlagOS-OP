# P800 运行时分阶段探测分析（2026-11-27）

> 本目录是诊断材料，不是 C1/C2 验收证据。原始结果绑定提交
> `cb62c806476bf783a49667c32b2c764d9a254b6b`，运行前工作树为空。

## 1. 复现环境

| 项 | 值 |
|---|---|
| profile | `p800-kunlunxin-gems5` |
| 容器镜像 ID | `e180a0dd4267` |
| 容器 | `liyanz-flaggems-p800` |
| 物理设备 | XPU 1/2（容器内逻辑 `cuda:0/1`） |
| 代码提交 | `cb62c806476bf783a49667c32b2c764d9a254b6b` |
| 探测命令 | `python ops/sigmoid/script/p800_runtime_probe.py --out-dir ops/sigmoid/evidence/p800-runtime-failure-2026-11-27` |

完整命令、环境变量、设备快照、进程快照、PID、时间和文件摘要见
[manifest.json](manifest.json)。汇总见 [REPORT.md](REPORT.md)。

## 2. 分阶段结果与归因

| 阶段 | 结果 | 解释 |
|---|---|---|
| 环境身份检查 | PASS | profile、提交和运行环境可读取 |
| device copy：17 / 255 / 14336 | PASS | 基础设备分配和数据传输正常 |
| native sigmoid：17 / 255 / 14336 | PASS | PyTorch 原生 sigmoid 路径正常 |
| generic Triton：257 | **INVALID_PROBE** | 探针局部导入导致 JIT 解析不到 `tl`，是探针实现错误，不能据此判断 Triton 后端或服务器故障 |
| 仓库 sigmoid Triton：257 | PASS | 通用边界路径可运行 |
| 仓库 sigmoid Triton：255 | PASS | 255 专用边界路径可运行 |
| vLLM TP=2 C2 baseline | **ENVIRONMENT_FAIL** | 两名 worker 均在 `ncclCommInitRank` 报 `NCCL error: unhandled cuda error`，EngineCore 未启动 |

`generic_triton_257` 的原始堆栈位于
[generic_triton_257.stderr.log](generic_triton_257.stderr.log)，直接错误为
`NameError('tl is not defined')`。因此，清单中的该项 `FAIL` 仅表示子进程非零退出；
它在本分析中被降级为 **INVALID_PROBE**，不作为环境故障证据，也不参与算子验收判断。

vLLM 的直接错误位于 [vllm_tp2_c2.stdout.log](vllm_tp2_c2.stdout.log)：
两个 worker 均在 `GroupCoordinator -> PyNcclCommunicator -> ncclCommInitRank`
阶段失败。失败发生在模型生成、sigmoid 注入和命中计数之前，因此当前只能证明
TP=2 分布式初始化被环境阻断，不能据此判断算子 C2 行为通过或失败。

## 3. 完整性与边界

- `manifest.json` 记录的 22 份 stdout/stderr SHA-256 已逐一回读校验，全部匹配。
- 11 个阶段按计划各执行一次：9 个 PASS、1 个无效探测、1 个环境失败；无超时。
- 探测结束后未发现本工具、vLLM EngineCore 或 worker 残留进程。
- 257 与 255 的仓库 sigmoid Triton 最小用例通过，只说明这两个诊断点可运行，
  不替代尺寸、dtype、布局、黄金数据、三层测试和性能矩阵的完整验收。
- TP=2 故障需由平台侧检查 P800/XPU 的 NCCL/XCCL 兼容、设备通信和容器映射；
  在环境恢复前不重复运行 C2，也不把历史 C2 或历史性能结果表述为当前结论。

## 4. 可复核入口

- 原始机器清单：[manifest.json](manifest.json)
- 自动汇总：[REPORT.md](REPORT.md)
- TP=2 完整 stdout：[vllm_tp2_c2.stdout.log](vllm_tp2_c2.stdout.log)
- TP=2 完整 stderr：[vllm_tp2_c2.stderr.log](vllm_tp2_c2.stderr.log)
- 255 专用路径 stdout：[sigmoid_255.stdout.log](sigmoid_255.stdout.log)
- 257 通用路径 stdout：[sigmoid_257.stdout.log](sigmoid_257.stdout.log)
