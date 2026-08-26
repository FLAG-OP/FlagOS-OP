# 设备 Profile

[← 返回文档中心](index.md)

> 字段是[设备 profile](#fields)机制的核心，也是[黄金](testing.md#golden)
> 与[kernel 层](testing.md#kernel-level)跨设备可比的基础。

## 设计

芯片差异声明于 `configs/devices/<芯片名>.yaml`。字段速查:

| 字段 | 作用 |
|---|---|
| `vendor` | 与 flag_gems DeviceDetector().vendor_name 匹配（自动探测用） |
| `auto_detect.module` | 可 import 即识别该设备 |
| `device.torch_device` | 算子层测试设备串（cuda:1 / xpu:0 / npu:0 / cpu） |
| `device.visible_devices` | 框架级测试暴露的卡号 |
| `device.dispatch_key` | A1 aten 注册 key（CUDA / PrivateUse1 / CPU ...） |
| `golden_engine` | vllm 或 transformers（CPU 用后者） |
| `framework.*` | vLLM 引擎参数（TP/显存/序列数/模型路径/seed） |
| `framework.quirks.*` | 设备怪癖开关（见下） |
| `vendor_delegate` | audit 委托的厂商 kernel（"pkg.func"）；null→reference |
| `vendor_kernels` | [kernel 层](testing.md#kernel-level)直测的厂商 kernel 清单（含 `expected_ok: false` 负例与 `build: csrc` JIT 项） |

<a id="fields"></a>
## 字段速查

（见上表）

## quirks 说明

- `keep_default_prefer: true` — 保持 `VLLM_FL_PREFER` 继承值（避免覆盖）
  （参考实例 P800: 环境默认非法值恰好让算子回落 vendor kernel，是保护路径）
- `require_two_visible_devices: true` — 单卡路径有厂商 reshape_and_cache
  通道异常史，固定 TP=2

<a id="onboard"></a>
## 新芯片接入（4 步）

1. 复制 `_template.yaml` 为 `<芯片名>.yaml` 并填写字段
2. `python3 run.py --all --device <芯片名>` 跑 9 格矩阵 + consistency
3. 有厂商 C++ kernel 时按 [BUILD.md](../routes/b_vendor/csrc/BUILD.md) 编译接入，
   并在 `vendor_kernels` 声明（含[哨兵负例](testing.md#sentinel)）
4. 建议立即: 生成[黄金](testing.md#golden) + 跑一次[漂移实验](testing.md#drift)
   + 用[哨兵检查](testing.md#sentinel)建立你自己的[已知问题清单](known-issues.md)

## 自动探测顺序

1. 各 profile 的 `auto_detect.module` 可 import → 选中
2. flag_gems `DeviceDetector().vendor_name` 匹配 `vendor` 字段 → 选中
3. 失败则报错并列出可用 profile（提示 `--device` 显式指定）

## 参考实例

- `p800-kunlunxin`: 9/9 矩阵 + 跨层一致性全绿（本库首个完整验证 case，
  其设备特有问题见 known-issues.md）
- `nvidia`: profile 就绪；曾借 CUDA 兼容层在加速卡环境验证过
  profile 切换机制（真实 N 卡的 vLLM 引擎路径已备好）
- `cpu`: transformers 黄金引擎已实测（确定性语义参考）

其他芯片: 复制 `_template.yaml` 填写后即可进入同一体.
