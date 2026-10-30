# 基座镜像索引

[← 返回文档中心](index.md)

这一页索引各平台协作方提供的当前基座镜像、用途、分发状态和与 FlagOS-OP 环境锁的关系。
FlagOS-OP 不构建、不分发这些镜像，也不把镜像登记等同于算子支持完成。
本页由 `configs/images/catalog.yaml` 生成，修改后不要手工编辑。

外部索引来源：[FlagRT/runtime-team · dev/images/image_list.md](https://github.com/FlagRT/runtime-team/blob/74197bf7d4b5dc8cdd3b2b89b15ac3561936dda6/dev/images/image_list.md)，
revision `74197bf7d4b5dc8cdd3b2b89b15ac3561936dda6`，读取时间 2026-10-16。

维护分工：

- 基座构建与分发：FlagRT runtime-team / 各平台基座提供方。
- FlagOS-OP 记录内容：FlagOS-OP 只登记线索并维护算子验证环境锁。
- 边界：本 catalog 不表示 FlagOS-OP 已重建或验证镜像本身。

## 阅读边界

| 问题 | 判断位置 |
|---|---|
| 基座镜像 tag、用途、外部复现状态 | 本页 |
| FlagOS-OP 实测硬件、系统与软件栈 | [软硬件环境版本汇总](environments.md) |
| 算子 / 平台是否已有交付证据 | [算子 / 平台矩阵](operator-catalog.md) |
| 镜像构建过程、digest、已知限制 | 外部索引和其锁文件 |

`version-aligned` 只表示包版本线一致。FlagOS-OP 环境锁未登记 image id 或 digest 时，
不能宣称算子验证发生在同一镜像实例上。

<a id="current"></a>
## 当前索引

| 平台 | 用途 / 系列 | 镜像引用 | 外部状态 | 分发 | 与 FlagOS-OP 的关系 |
|---|---|---|---|---|---|
| 华为昇腾 | 昇腾 910C 算子运行时基座<br>`ascend-operator-runtime` / `v3` | `flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64` | 部分可复现 | 内部构建 | 未接入 FlagOS-OP 环境锁 |
| 华为昇腾 | 昇腾 910C 训练腿<br>`ascend-operator-runtime-comm` / `v3` | `flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64` | 部分可复现 | 离线发布 | 未接入 FlagOS-OP 环境锁 |
| 华为昇腾 | 昇腾 910C 推理腿<br>`ascend-infer-vllm` / `v1` | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` | 登记式基座 | 公开 registry | 未接入 FlagOS-OP 环境锁 |
| 昆仑芯 | 昆仑芯 P800 算子运行时基座<br>`kunlun-operator-runtime` / `v2` | `flagrt/kunlun-operator-runtime:2.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64` | 外部可复现构建 | 内部构建 | 版本对齐；未声明同镜像实例 |
| 平头哥 | 平头哥 PPU 运行时基座<br>`thead-ppu-runtime` / `v1` | `harbor.baai.ac.cn/flagtree/flagtree-ppu-py312-torch2.10.0-sdk2.1.0-cu130-ubuntu24.04:202607-3.6-base` | 登记式基座 | 公开 registry | 版本对齐；未声明同镜像实例 |
| 寒武纪 | 寒武纪 MLU590 运行时基座<br>`cambricon-mlu-runtime` / `v1` | `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0` | 登记式基座 | 公开 registry | 未接入 FlagOS-OP 环境锁 |

### 平台说明

<a id="ascend-910c-operator-runtime-v3"></a>
#### 昇腾 910C 算子运行时基座

- 内部构建层，不对外发布。
- FlagOS-OP 当前 ascend910 环境未登记该镜像 tag，不声明同镜像验证。
- 关联环境锁：未登记。

<a id="ascend-910c-operator-runtime-comm-v3"></a>
#### 昇腾 910C 训练腿

- 外部源标记为训练腿发布物，带 4 条已知限制。
- FlagOS-OP 当前算子环境未声明训练腿验证。
- 关联环境锁：未登记。

<a id="ascend-910c-infer-vllm-v1"></a>
#### 昇腾 910C 推理腿

- 华为昇腾官方镜像，外部源按 digest 锁定。
- FlagOS-OP 当前 ascend910 环境是 kernel/op/golden/perf 交付，未声明该推理镜像。
- 关联环境锁：未登记。

<a id="kunlunxin-p800-operator-runtime-v2"></a>
#### 昆仑芯 P800 算子运行时基座

- 外部源标记为可用配方重建并实机自检通过。
- 与 FlagOS-OP p800-kunlunxin 环境锁的 Python/torch/FlagGems 主版本线对齐。
- 该关联限于 FlagGems 4.2 基线；p800-kunlunxin-gems5 的 5.0 dev 复测容器尚未登记基座镜像，不应由本 tag 推断其构建来源。
- FlagOS-OP 未登记运行容器的 image id 或 digest，因此不声明“同镜像实例已验证”。
- 关联环境锁：`p800-kunlunxin`。

<a id="thead-ppu-runtime-v1"></a>
#### 平头哥 PPU 运行时基座

- 外部源标记为 FlagTree 官方镜像并按 digest 锁定。
- 与 FlagOS-OP thead-ppu 环境锁的 Python/torch/triton 主版本线对齐（py3.12 / torch 2.10.0 / triton 3.6.0）。
- FlagOS-OP 未登记运行容器的 image id 或 digest，因此不声明“同镜像实例已验证”。
- 关联环境锁：`thead-ppu`。

<a id="cambricon-mlu-runtime-v1"></a>
#### 寒武纪 MLU590 运行时基座

- 外部源标记为 FlagOS 官方运行时镜像并按 digest 锁定。
- 该镜像线为 Neuware 4.4.3；FlagOS-OP 当前 cambricon/mlu590 环境为 Neuware 4.7.6 或 torch_mlu 1.33+，不能视为同一运行环境。
- 关联环境锁：未登记。

<a id="coverage"></a>
## 平台覆盖

| 平台家族 | 外部基座线索 | FlagOS-OP 算子矩阵 |
|---|---|---|
| 华为昇腾 | `ascend-operator-runtime` · `ascend-operator-runtime-comm` · `ascend-infer-vllm` | 部分交付 |
| 寒武纪 | `cambricon-mlu-runtime` | 部分交付 |
| 昆仑芯 | `kunlun-operator-runtime` | 部分交付 |
| 平头哥 | `thead-ppu-runtime` | 部分交付 |
| 燧原 GCU | 未登记 | 规划中 |
| 壁仞 | 未登记 | 规划中 |

<a id="extend"></a>
## 平台方扩展方式

1. 在 `configs/images/catalog.yaml` 的 `images` 列表追加一行，不要手工编辑本页。
2. 填写平台家族、用途、镜像引用、外部状态、分发方式和备注。
3. 如果该镜像只是版本对齐，`flagos_op_relation` 使用 `version-aligned`，并在备注中写明对齐项。
4. `version-aligned` 表示包版本对齐，但环境锁尚未证明使用同一镜像实例。
5. 只有环境锁登记 image id 或 digest 并完成算子验证时，才使用 `validated-in-image`。
6. 运行生成与校验命令后提交 PR：

```bash
python3 scripts/build_base_image_index.py
python3 scripts/build_base_image_index.py --check
```

常用字段：

| 字段 | 含义 |
|---|---|
| `family` / `series` / `version` | 平台家族、镜像系列和版本号 |
| `registry_reference` | 镜像 tag；digest 继续维护在外部基座仓库 |
| `status` | 外部源声明的 `registered` / `functional-repro` / `partial-repro` |
| `distribution` | 公开 registry、内部构建或离线发布 |
| `current` | 是否当前推荐入口；同一平台同一系列只能有一个当前版本 |
| `superseded_by` / `archival_reason` | 历史版本的后继镜像或归档原因 |
| `related_profiles` | 已建立对应 FlagOS-OP 环境锁的 profile |
| `flagos_op_relation` | 未接入、版本对齐、镜像内验证或平台仍为 planned |

登记规则：

- 不用 planned 平台预填镜像来推断算子支持完成；
- 不用包版本相同推断镜像实例相同；
- 不把外部 `functional-repro` 转化为 FlagOS-OP 的算子验证结论；
- 镜像 digest、构建配方和已知限制维护在外部基座仓库；本仓库只保留索引和关联关系。

<!-- AUTO-GENERATED FROM configs/images/catalog.yaml; do not edit. -->
