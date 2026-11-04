# C2 EngineCore 采集与验收（Ascend / vllm-plugin-fl 栈）

真实 vLLM **EngineCore（多进程）** 双跑 + 生成阶段计数的**完整可复现**采集链。
区别于 `ops/*/script/c2_real_vllm_ascend.py`（#44，vLLM-Ascend 栈），本链测试的是
**vllm-plugin-fl / PlatformFL** 栈。

## 目录

| 文件 | 作用 |
|---|---|
| `sitecustomize.py` | 在 vLLM EngineCore 子进程内注册目标 A1 算子（显式 platform），并 patch `vllm_fl.worker.worker.WorkerFL.execute_model`，对每次前向取 `CALL_COUNT` **前/后快照** → 得到生成阶段增量；进程退出时一次性 dump（避免热路径 I/O）。 |
| `make_model.py` | **确定性**构造参考模型（`LlamaConfig(hidden=64, heads=4, layers=2, head_dim=16)` + 固定 WordLevel 词表 + `torch.manual_seed(0)`）。模型身份由本文件复现。 |
| `run.py` | 单次 EngineCore 进程：`--mode init`（只初始化、不 generate）/ `--mode full`（生成）。设置 `PYTHONPATH`/`FLAGOS_C2_*` 后再 import vllm，使 spawn 子进程加载本 `sitecustomize.py`。 |
| `validate.py` | 验收入口：跑 baseline / init-only / injected 三次，聚合 per-PID dump，`decide()` 判定（非零退出即失败）。 |

## 环境（本次实测）

- driver 25.5.0 / CANN 9.0.0 / SOC Ascend910_9382
- torch 2.10.0+cpu / torch_npu 2.10.0 / flagtree(triton) 3.5.1
- vllm 0.20.2+empty / vllm-plugin-fl `0.0.0+g8b059122e`
- 可见设备固定：`ASCEND_RT_VISIBLE_DEVICES=1`

## 复现命令

```bash
# 1) 构建确定性模型（身份可复现）
python3 scripts/c2_enginecore/make_model.py /tmp/c2model

# 2) 采集 + 验收（退出码 0 = 通过，非 0 = 失败）
ASCEND_RT_VISIBLE_DEVICES=1 \
python3 scripts/c2_enginecore/validate.py \
  --op empty_strided --model /tmp/c2model --outdir /tmp/c2out --gpu-util 0.5

# 3) 归档到算子证据目录
cp -r /tmp/c2out/empty_strided/. ops/empty_strided/reports/c2_ascend910/
```

## 判定（`validate.decide`，失败返回非零）

- 三次运行 `rc == 0`；
- baseline / injected token 均非空且**逐位一致**；
- 存在 EngineCore 阶段 dump（观察到 `execute_model`）；
- **生成阶段命中 > 0**（仅 init 命中不算）。

负例（init-only、零生成命中、空输出、输出失配、子进程失败、缺阶段 dump）由
`tests/unit/test_c2_enginecore_decide.py` 覆盖。

## 证据产物

`ops/<op>/reports/c2_ascend910/` 下：
- `summary.json`（run_id、env、ids、init_hits/generation_hits/final_hits、logs、passed）
- `baseline.stdout.log` / `init.stdout.log` / `inject.stdout.log`
- `counts_*/count_<pid>.json`（每个 EngineCore/主进程的 init/generation/final 计数）
- `run_id.txt`、`commands.txt`
