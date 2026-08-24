# FlagOS 算子开发模板库（矩阵化 · 芯片泛化）

基于: FlagGems 4.2.1rc0 / vllm-plugin-FL 0.1.0 / vLLM 0.13.0 / Triton
默认 case: 2×P800 (Kunlunxin XPU)，其他芯片通过设备 profile 接入。

## 矩阵总览

**本机 2×P800 实测: 6/6 全部通过**（含 6 次真实 vLLM 推理）。

| | 算子层 (op) | 框架层 (framework) |
|---|---|---|
| **A1** Triton→aten dispatcher | `--route a1 --level op`<br>gelu 被真实拦截+精度 | `--route a1 --level framework`<br>aten::silu 恒等计数注入真实 vLLM |
| **A2** Triton→FlagOS dispatch | `--route a2 --level op`<br>三段式: 精度/分发/性能 | `--route a2 --level framework`<br>Triton silu_and_mul 经 vendor:triton-template 注入 |
| **B** 厂商语言→vendor backend | `--route b --level op`<br>注册/选择/计数/语义 | `--route b --level framework`<br>audit vendor 拦截真实 vLLM |

## 快速开始

```bash
# 全矩阵（6 格，本机 P800 约 15-20 分钟，含 6 次真实 vLLM 推理）
./scripts/run_all.sh

# 单格
python3 run.py --route a1 --level op --device p800-kunlunxin

# 薄封装（等价单格; DEVICE 环境变量切芯片）
DEVICE=p800-kunlunxin ./scripts/run_b_fw.sh

# 列出设备 profile 与矩阵
python3 run.py --list
```

## 目录结构

```
flagos-op-templates/
├── run.py                    统一矩阵入口
├── configs/devices/          设备 profile（p800-kunlunxin / nvidia / _template）
├── docs/                     中文文档（入门/架构/路线/测试/泛化/已知问题）
├── common/                   device.py(profile加载+探测) / ref_impls.py
├── routes/
│   ├── a1_aten/              Triton → torch dispatcher
│   ├── a2_dispatch/          Triton → FlagOS dispatch 插件
│   └── b_vendor/             厂商语言 → vendor backend（csrc + audit）
├── examples/                 6 个矩阵格的可运行样例（算子层自包含）
├── tests/
│   ├── op_level/             算子层测试（每路线一个，签名 run(profile))
│   └── framework_level/      框架层测试（真实 vLLM 推理验证）
├── scripts/                  run_all.sh + 6 个薄封装
├── golden/                   黄金输出（多快照+共识前缀，回归锚点）
└── results/                  每格结果 JSON（gitignore）
```

## 文档

完整中文文档在 [docs/](docs/index.md): 快速开始 / 体系结构 /
三条路线详解 / 测试体系 / 设备接入 / 已知问题清单。
每格样例见 [examples/](examples/README.md)。

## 新芯片接入（3 步）

1. 复制 `configs/devices/_template.yaml` 为 `<芯片名>.yaml`，填写：
   vendor / torch_device / visible_devices / dispatch_key /
   framework 引擎参数 / vendor_delegate（无厂商库填 null）
2. `python3 run.py --all --device <芯片名>` 跑 6 格矩阵
3. 有厂商 C++ kernel 时替换 `routes/b_vendor/csrc/` 并按 BUILD.md 编译

> 泛化已验证: `--device nvidia` profile（cuda:0）在本机经 CUDA 兼容层
> 同样跑通 op 级格子——同一套代码、不同 profile、不同物理设备。

## 设备 profile 关键字段

| 字段 | 作用 |
|---|---|
| `device.torch_device` | 算子层测试设备串（cuda:1 / xpu:0 / npu:0 ...） |
| `device.dispatch_key` | A1 aten 注册 key（CUDA / PrivateUse1 ...） |
| `framework.*` | 框架级 vLLM 引擎参数 + quirks |
| `framework.quirks.keep_default_prefer` | P800 保护路径：禁止覆盖 VLLM_FL_PREFER |
| `vendor_delegate` | audit 委托的厂商 kernel（"pkg.func"）；null→reference |

## 本机已知注意事项

1. **不要设 `VLLM_FL_PREFER=flagos`**（P800）: FlagGems 的 rms_norm Triton
   dispatch 实现在真实 vLLM 前向中会失败；容器默认非法值恰好让所有
   算子回落 vendor.kunlunxin，是生产可用路径。框架级测试只用
   `VLLM_FL_PER_OP` 精确钉住目标算子。
2. **插件入口函数名**: 代码实际查找 `register` / `vllm_fl_register`
   （README 文档写的 `register_builtins` 仅适用内置 backend）。
3. **P800 单卡路径** 曾出现厂商 reshape_and_cache 通道异常，
   profile 固定 TP=2。
4. **框架层输出断言分两种**: 数值恒等路径（aten 恒等覆盖 / audit
   reference 委托，基线与插件同钉 reference）要求前 8 token 完全一致
   （实测本栈跨进程 16+ token 偶发非确定性，全量一致不可靠）；自定义
   数值实现（Triton）允许混沌分叉，用前 2 token 一致率断言。
5. **⚠️ 厂商 kernel 已知问题**: `xtorch_ops.swiglu` 在本机独立 eager
   调用下**不写输出张量**（哨兵测试: torch.full 预填后调用，全部元素
   保持原值；返回 int 0）。即 vendor.kunlunxin 的 silu_and_mul 实现在
   此栈上不可靠——生产 vLLM 表面正常疑似依赖 allocator 复用旧激活
   内存。故 B 框架级测试使用 reference 委托，厂商委托仅保留手动验证
   入口。此问题建议向芯片厂商反馈。
6. **测试输入模板化**: `inputs/spec.yaml` 声明式定义输入
   （fixed/length_sweep/edge_cases 文本类 + token_based 算子类），
   `scripts/gen_inputs.py` 生成 `prompts.txt` 与 `token_inputs.json`，
   同一 spec 在任何硬件生成完全相同的输入（黄金可比的前提）。
7. **跨设备黄金输出**: `scripts/build_golden.py` 按设备生成黄金
   （多快照+多数票共识前缀，`golden/<device>_golden.json` 应入库）:
   - `--device cpu` → HuggingFace transformers 直接 CPU 推理
     （权威语义参考，确定性，完全绕过 vLLM/厂商栈）
   - `--device nvidia` → vLLM 引擎（需真实 N 卡，profile 已就绪）
   - `--device p800-kunlunxin` → vLLM reference 路径 ×3 快照
   `scripts/compare_golden.py --devices p800-kunlunxin,cpu` 输出跨设备
   前缀一致矩阵。A1/B 框架测试自动比对同设备黄金；exact 断言前缀
   按黄金实测稳定前缀**自适应**（不人工拍 8）。

8. **⚠️ embedding 静默越界**: 随机模型 tokenizer 词表(151669)大于
   embedding(128256)，文本输入产生越界 token id——vLLM/XPU 路径的
   embedding 查找**不做边界检查**（静默读越界内存），CPU transformers
   则正确报 IndexError。模板已统一为 tokenize 后按词表钳制的 token
   输入（`PROMPTS_AS_TOKENS=1`），双端输入一致且合法。

9. **漂移确认实验结论**（`scripts/drift_study.py`，结果在
   `results/drift_*.json`）: 受控顺序条件下，reference 路径 8 连跑 +
   vendor 路径 4 连跑（各 6 prompt × 32 token）**全部 100% 确定**；
   此前观察到的偶发后期 token 漂移为低概率条件相关事件。防御性设计
   （前 8 token 断言 + 黄金多数票共识）已覆盖残余风险。
