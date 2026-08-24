# 设备 Profile 与芯片泛化

## 设计

芯片差异全部收敛到 `configs/devices/<芯片名>.yaml`，
测试代码零硬编码。字段速查:

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

## quirks 说明

- `keep_default_prefer: true` — 禁止覆盖 `VLLM_FL_PREFER`
  （P800: 容器默认非法值恰好让算子回落 vendor kernel，是保护路径）
- `require_two_visible_devices: true` — 单卡路径有厂商 reshape_and_cache
  通道异常史，固定 TP=2

## 新芯片接入（3 步）

1. 复制 `_template.yaml` 为 `<芯片名>.yaml` 并填写字段
2. `python3 run.py --all --device <芯片名>` 跑 6 格矩阵
3. 有厂商 C++ kernel 时按 `routes/b_vendor/csrc/BUILD.md` 编译接入

## 自动探测顺序

1. 各 profile 的 `auto_detect.module` 可 import → 选中
2. flag_gems `DeviceDetector().vendor_name` 匹配 `vendor` 字段 → 选中
3. 失败则报错并列出可用 profile（提示 `--device` 显式指定）

## 已验证

- `p800-kunlunxin`: 6/6 矩阵全绿（本库的主 case）
- `nvidia`: profile 就绪；在 P800 机器上经 CUDA 兼容层映射可跑
  （真实 N 卡环境下 vLLM 引擎路径已备好）
- `cpu`: transformers 黄金引擎已实测（确定性，跨设备锚点）
