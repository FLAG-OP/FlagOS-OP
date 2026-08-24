# FlagOS 算子开发模板库（矩阵化 · 芯片泛化）

一套**设备无关**的 FlagOS 自定义算子开发与验证模板：三条实现路线 ×
三层验证层级构成 9 格矩阵，芯片差异全部收敛到设备 profile。
任何 AI 加速卡接入只需写一份 YAML，测试代码零硬编码。

- 依赖形态: FlagGems + vllm-plugin-FL + vLLM + Triton（版本随环境，
  参考验证环境: FlagGems 4.2.1rc0 / vllm-plugin-FL 0.1.0 / vLLM 0.13.0）
- **参考实例**: `configs/devices/p800-kunlunxin.yaml`（2×P800，
  本库 9/9 矩阵 + 跨层一致性的首个完整验证 case）；`cpu.yaml` /
  `nvidia.yaml` 为另外两个内置 profile

## 矩阵总览（3 路线 × 3 层级 = 9 格）

| | kernel 层（硬件语言直测） | op 层（注册/分发/拦截） | framework 层（真实推理） |
|---|---|---|---|
| **A1** Triton→aten | Triton kernel 直测 | aten 注册+拦截 | aten 恒等计数注入 vLLM |
| **A2** Triton→dispatch | Triton kernel 直测 | dispatch 注册+策略切换 | vendor 身份注入 |
| **B** 厂商语言→vendor | **厂商 kernel 直测+C++ JIT 编译闭环+哨兵检查** | vendor 注册/选择 | audit vendor 拦截 |

另有 `--consistency`: 同算子 L0 直调 ↔ L2 dispatch 张量级一致矩阵。

## 快速开始

```bash
# 全矩阵（9 格 + consistency；时长视设备而定，GPU case 约 20 分钟）
DEVICE=<profile名> ./scripts/run_all.sh

# 单格（示例用参考 profile，替换为你的芯片即可）
python3 run.py --route a1 --level op --device p800-kunlunxin

# 列出可用设备 profile 与矩阵
python3 run.py --list
```

## 目录结构

```
flagos-op-templates/
├── run.py                    统一矩阵入口
├── configs/devices/          设备 profile（参考实例 p800 + cpu/nvidia/_template）
├── docs/                     中文文档（入门/架构/路线/测试/泛化/已知问题）
├── common/                   设备抽象 / kernel spec / 输入模板 / 参考实现
├── routes/                   三条路线正式实现
│   ├── a1_aten/              Triton → torch dispatcher
│   ├── a2_dispatch/          Triton → FlagOS dispatch 插件
│   └── b_vendor/             厂商语言 → vendor backend（csrc + audit）
├── examples/                 9 个矩阵格的可运行样例（kernel/op 层自包含）
├── tests/
│   ├── kernel_level/         kernel 直测层 + 跨层一致性
│   ├── op_level/             注册/分发/拦截层
│   └── framework_level/      真实推理层
├── golden/                   黄金输出（多快照+共识前缀，回归锚点）
└── scripts/                  一键脚本 / 黄金构建 / 漂移实验 / 输入生成 / 报告
```

## 文档

完整中文文档在 [docs/](docs/index.md): 快速开始 / 体系结构 /
三条路线详解 / 测试体系 / 设备接入 / 已知问题。
每格样例见 [examples/](examples/README.md)。

## 新芯片接入（3 步）

1. 复制 `configs/devices/_template.yaml` 为 `<芯片名>.yaml`，填写：
   vendor / torch_device / visible_devices / dispatch_key /
   framework 引擎参数 / vendor_delegate / vendor_kernels 清单
2. `python3 run.py --all --device <芯片名>` 跑 9 格矩阵 + consistency
3. 有厂商 C++ kernel 时按 `routes/b_vendor/csrc/BUILD.md` 编译接入

## 设备 profile 关键字段

| 字段 | 作用 |
|---|---|
| `device.torch_device` | kernel/op 层测试设备串（cuda:0 / xpu:1 / npu:0 ...） |
| `device.dispatch_key` | A1 aten 注册 key（CUDA / PrivateUse1 ...） |
| `framework.*` | 框架级 vLLM 引擎参数 + quirks（设备怪癖开关） |
| `vendor_delegate` | audit 委托的厂商 kernel（"pkg.func"）；null→reference |
| `vendor_kernels` | kernel 层直测的厂商 kernel 声明清单（含负例标注） |

## 通用注意事项（适用所有设备栈）

1. **插件入口函数名**: dispatch 代码实际查找 `register` /
   `vllm_fl_register`（官方文档写的 `register_builtins` 仅适用内置
   backend）。
2. **框架层输出断言分两种**: 数值恒等路径（aten 恒等覆盖 / audit
   reference 委托）要求逐 prompt 前缀完全一致，前缀长度按该设备黄金
   实测稳定长度**自适应**；自定义数值实现允许混沌分叉（随机权重+
   贪心解码会放大数值微差），用前 2 token 一致率断言。
3. **测试输入模板化**: `inputs/spec.yaml` 声明式定义输入，
   同一 spec 在任何硬件生成完全相同的输入——跨设备黄金可比的前提。
4. **跨设备黄金输出**: `build_golden.py` 按设备生成黄金（CPU 经
   HuggingFace transformers 为权威语义参考；加速卡走 vLLM），
   `compare_golden.py` 输出跨设备前缀一致矩阵。
5. **性能基准用短采样**（≤100 次）: 逐次新分配大输出的长循环会触发
   分配器池增长，均值失真一两个数量级。
6. **vLLM v1 前向在子进程**: 主进程 torch.library 注册不传播，
   aten 路线框架级需 sitecustomize 注入桥（本库已内置）。

## 参考实例的实测记录

P800 case 的设备特有问题（厂商 kernel 缺陷 / 栈怪癖 / 数值行为）全部
沉淀在 [docs/known-issues.md](docs/known-issues.md)——每接入一种新芯片，
建议用同样方法（哨兵检查 / drift_study / 黄金比对）建立该设备自己的
问题清单。通用机制能抓到的问题类型:

- 厂商 kernel 不写输出（哨兵检查）
- kernel 按物理设备分化的正确性差异（逐设备直测）
- 栈升级导致的数值行为漂移（黄金回归）
- 跨进程非确定性（漂移实验）
