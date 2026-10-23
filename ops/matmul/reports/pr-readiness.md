# matmul：PR 自查与合并后回归

<!-- tyf-2026-10-23-fix; tyf <tyanfeng@mail.ustc.edu.cn> -->

日期：2026-10-23。开发者：陶焱烽；Git 作者身份：
`A1tria7 <a1tria1003@gmail.com>`。

状态：**PARTIAL**。本阶段目标是 A1 功能、正确性、C1 最小消费者和初始
性能记录。性能门禁仍保留 `4 FAIL / 4 WARN`，不声明完整交付验收通过。
本次不扩展 C2 真实引擎、不升级到 FlagGems 5.0，也不调整性能基线。

## 自查范围

| 项目 | 状态 | 证据 / 范围 |
|---|---|---|
| 核心八件 | PASS | README、REPORT、包入口、reference、register、example、kernel、test 齐全 |
| 算子目录 | PASS | 已登记 `ops/catalog.yaml`，生成 `ops/README.md`，目录一致性检查通过 |
| 包化导入 | PASS | 按 CI 扫描全部 17 个算子；屏蔽 Triton、FlagGems、torch_npu、torch_mlu 后仍全部通过 |
| 包模式 / 脚本模式回归 | PASS | 两种入口的 torch backend 广播前向、双输入梯度及缺少 Triton 的错误信息通过 |
| 仓库单元测试 | PASS | `tests/unit`：49 passed；本机临时环境，仅 CPU 检查 |
| 环境 / 镜像索引 | PASS | `build_env_index.py --check`、`build_base_image_index.py --check` |
| 文档 / 语法 / 配置 / 基线结构 | PASS | 按 CI 脚本检查提交文件与本次新增文件快照，均 0 错误 |
| intake 契约 | PASS | `intake_validate.py --validate-only`：2 case，0 错误 |
| 目标设备功能回归 | PASS | XPU 2：黄金直调/A1 各 144/144；三级及 FlagGems 共存通过 |
| 性能对照 | PASS | native/Triton 前后向及三 dtype 记录、报告、144 条基线齐全 |
| 性能门禁 | PARTIAL | 历史完整复测 136 OK / 4 WARN / 4 FAIL；自研 72 条全部 OK |
| 多平台文档 | 不适用 | 当前只登记昆仑芯实测；其他平台正式适配时补 PLATFORM / MERGE |
| 署名 | 已标注，提交时复核 | 改动标记格式 `tyf-2026-10-23-fix` 与本人邮箱；Git 作者保持 `A1tria7` |

`ops/test_packaging.py` 完整脚本依赖 Ascend NPU，并包含固定路径；本次
包化 PASS 指 `.github/workflows/ci.yml` 的 CPU 导入扫描，不声称跑过 NPU
双算子消费段。仓库单元测试通过也不能替代 P800 的设备实测。

## 本次改动

1. `register.py` 在选择 backend 时才加载具体实现。查看 `register_a1`
   导出不再要求安装 Triton；每次注册的前后向函数绑定在该注册闭包中。
   显式区分包与脚本路径，保留真实缺失依赖异常。
2. 登记 matmul 的 P800 范围，明确 `PARTIAL` 和未闭合的性能门禁。
3. 补齐 C1 消费者字段和历史源码身份，按新规范区分历史证据与本次回归。
4. 修复环境检查对空版本元数据的格式化异常；缺失或不可读仍返回不符，
   不把未知版本判定为通过。

## P800 回归范围与身份

| 字段 | 值 |
|---|---|
| consumer_type | `MatmulConsumer`：两次 matmul + FlagGems GELU，含双输入及权重反向 |
| 证据等级 | C1 最小消费者 |
| 引擎 / 版本 | PyTorch eager 2.9.0+cu129；FlagGems 4.2.1rc0 |
| 编译器 | Triton 3.0.0+03c4c9be，`XMLIR_MATMUL_FAST_MODE=0` |
| profile | `p800-kunlunxin`，物理 XPU 2，`CUDA_VISIBLE_DEVICES=1,2` → `cuda:1` |
| 测试日期 | 2026-10-23 |
| 基础 commit | `ab601cf` 加本次未提交改动；不是纯净 commit 的测试结果 |
| 运行隔离 | 个人容器、UID 1030、独立源码目录和 Triton 缓存；逐测试进程执行 |

环境核对：个人容器 root 身份只读检查全部版本与锁一致。普通 UID 查询
部分厂商包元数据返回空值；已修正检查工具，使其报告不符。未改动厂商包
权限或安装版本。功能测试使用普通 UID，验证实际运行能力。

设备源码经远端 SHA256 与本机比对一致（下表路径均相对于 `ops/matmul/`）：

| 文件 | SHA256 |
|---|---|
| `register.py` | `848ed851f06e4b1de82ecd2c9c6d4f7f86f999063e3b7caf7e9fd32ffee06e07` |
| `kernel/triton_level.py` | `fe160e9251b71bafda052098427dbe4398679badc1e86e1d173adad40a5d6228` |
| `test/framework_level.py` | `e7cff6e41979849638f45e1a2f2882deb1e823a54e827a53a2f4a0e46b6f4881` |

| 本次执行 | 结果 |
|---|---|
| CPU 黄金重建 | 144 组前向和双输入梯度 |
| kernel | 144 组；哨兵及零 stride 反向 PASS |
| K=1/2 SRAM 残留 | 36/36 PASS |
| Triton 黄金 | 144 pass / 0 fail，最坏相对误差 `7.75e-3` |
| A1 黄金 | 144 pass / 0 fail，144 次命中，最坏相对误差 `7.75e-3` |
| A1 op | 15 次命中，最坏原生相对误差 `1.087e-2` |
| C1 framework | 2 次命中；输出及三个梯度误差 `[0.001719, 0.003446, 0.003509, 0.002811]` |
| 一键三级 | kernel / op / framework 全部 PASS |
| FlagGems 共存 | 排除其既有 `copy_` 问题后 15 次命中，最坏相对误差 `1.087e-2` |

全部测试为独立进程、退出码 0。C1 功能结果可以标记 PASS；性能没有在本轮
重新采集，仍采用历史门禁结论。环境工具的普通 UID 复测正常返回退出码 1，
明确显示 XMLIR 元数据不可读；该结果不计入上述功能测试 PASS，也不掩盖为
环境检查通过。

## 复现入口

CPU 环境需 PyTorch、PyYAML、pytest；目标设备使用锁定的厂商运行时。

```bash
python3 scripts/build_ops_index.py --check
python3 -m pytest tests/unit -q
python3 scripts/check_env.py --device p800-kunlunxin
python3 ops/matmul/script/gen_golden.py
CUDA_VISIBLE_DEVICES=1,2 XMLIR_MATMUL_FAST_MODE=0 python3 ops/matmul/example.py p800-kunlunxin
CUDA_VISIBLE_DEVICES=1,2 XMLIR_MATMUL_FAST_MODE=0 python3 ops/matmul/script/check_accuracy.py --impl triton --device p800-kunlunxin
CUDA_VISIBLE_DEVICES=1,2 XMLIR_MATMUL_FAST_MODE=0 python3 ops/matmul/script/check_accuracy.py --impl a1-triton --device p800-kunlunxin
CUDA_VISIBLE_DEVICES=1,2 XMLIR_MATMUL_FAST_MODE=0 python3 ops/matmul/test/op_level.py p800-kunlunxin --with-flaggems
```

运行设备测试前确认目标卡空闲，并为本轮设置独立 `TRITON_CACHE_DIR`。
性能遗留保持 [性能报告](performance.md) 的原始结论，待后续性能阶段处理。
