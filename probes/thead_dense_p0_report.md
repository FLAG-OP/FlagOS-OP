# thead-ppu 稠密组 P0 探针取证报告（issue #55/#56/#57 前置）

- 日期: 2026-11-25（#56 证据补充: 同日）
- 设备: PPU-ZW810E ×16（真武 810E, ai-server 容器; 测试卡 6, 跑前 ppu-smi 空闲）
- 环境: `flagos-ppu-wt-dev` 基座（torch 2.10.0 / triton 3.6.0 / flagtree
  0.6.1+ppu3.6 [wheel 损坏在册] / vllm 0.19.0+cu130 / python 3.12.3;
  无 flag_gems——与 configs/env/thead-ppu.lock.yaml 一致）
- 探针: `probes/thead_dense_p0_probe.py`（`--isolated` 逐项子进程隔离;
  原始 JSON: `probes/thead_dense_p0_probe.json`）
- 复现: `PPU_PROBE_DEVICE=cuda:6 python3 probes/thead_dense_p0_probe.py --isolated`

## 一、issue #55 七算子原生可用性（真机）

| 算子 | 原生可用 | 精度（对 CPU fp32/独立参考） | 生产路线裁定 |
|---|---|---|---|
| `aten::linear` | ✅ | 2D+bias fp16 err=0.0; 3D bf16 no-bias err 1.5e-5 | 复用（native 委托） |
| `aten::layer_norm` | ✅ | fp16 err=0.0 | 复用（native 委托） |
| `aten::native_layer_norm` | ✅ | fp16 err 9.5e-7; 3 返回值齐备 | 复用（native 委托） |
| `aten::matmul` | ✅ | fp16 abs 3.9e-3（GEMM 判卷 5e-2 rel 内）; bf16 广播 err=0.0 | 复用（native 委托） |
| `aten::softmax` | ✅ | 公开 API err=0.0 | 复用（`softmax.int` 重载委托） |
| `aten::_softmax` | ✅ | err=0.0; h2f 输出 fp32 err 1.8e-7 | 复用（原语委托） |
| `aten::gelu` | ✅ | none 6.1e-5 / tanh 9.5e-7（fp16） | **自研 Triton**（见下） |

gelu 自研裁定依据: gelu 为逐 elementwise, 不踩 PPU Triton `tl.dot`
精度降级（sdpa §3c）与 #15a 尾块归约坑; 真机三层验收自研 Triton
**1.088x 原生**（0.131ms vs 0.143ms @ 8192² fp16, 2.05 TB/s）且携带
解析反向（AutogradCUDA 范式）。原生路径保留为对照与回退底座。

### 平台绑定结论（全部进各算子 PLATFORM.md）

1. **redispatch 键口径**: 委托实现的"借道原生"必须用**后端键 CUDA**
   redispatch——`CompositeExplicitAutograd` 落到 ATen 组合分解，与厂商
   kernel 累加序不同，半精度数值分化（bf16 layer_norm 差 **0.031** 实证;
   fp32 亦有 4.8e-7 差异）。CUDA 键与直调**逐位一致**（三 dtype 实证）。
2. **注册键口径**: `AutogradCUDA`（thead 范式, 自带反向）:
   - CUDA 基键注册会**覆盖**该栈挂在 CUDA 键上的 autocast 粘合
     （autocast_mode.cpp:169 同键覆盖告警实证）→ autograd 断链;
   - CUDA 键注册后 CUDA 键 redispatch **重入自身**（RecursionError 实证）;
   - AutogradCUDA 注册 + autograd.Function（forward=CUDA redispatch 下探
     完好厂商 kernel, backward=原生 *_backward 原语/解析式）全绿实证
     （七算子三层全过, framework 恒等/梯度对拍 diff=0）。
   - 平台边界: inference_mode 下 AutogradCUDA 零命中（known-issues #19.3）。
3. **PPU Triton 要点**: `tl.math.tanh` 不存在（erf 存在）→ 手写稳定
   tanh `1-2/(exp(2|x|)+1)`（known-issues #13 同款口径, IEEE inf 算术
   溢出安全）; `aten::gelu` 的 `approximate` 为 schema keyword-only
   （位置传参在 C++ boxed 层同样报 schema 不匹配）。
4. **native_layer_norm 形状平台分化**: mean/rstd 返回 **keepdim 形状**
   且恒 fp32（CPU 语义为去维形状）——值广播等价, 测试按 reshape 对拍。
5. **h2f 边界**: `_softmax(half_to_float=True)` 仅支持 Half 输入
   （bf16/fp32 报 "conversion is supported for Half type only"）——
   bf16/fp32 的 h2f 回退 fp32 域计算 + cast（数学等价）。
6. **softmax 重载集**: 本栈为 `int/Dimname/int_out`（**无 int_dtype**）;
   `Library.impl` 必须按 `softmax.int` 重载名绑定——按裸名 `softmax`
   绑定注册到无 schema 空壳, 调用永不命中（op 层实证）。

## 二、issue #56: `aten::_efficient_attention_forward` 缺口证据

**认领信息（本单要求回报项 1）**: 本单由 FlagOS-OP thead-ppu 线认领;
智源侧在途实现对齐项（关联分支/PR、目标版本）**待智源侧回复**——
上游 PR flagos-ai/FlagGems#1700 已关闭（合并提交承接）, 需要智源侧
确认承接提交与目标版本。

**布局修正与证据状态（复审更新）**: 复审指出首轮探针把 `[B,H,S,D]`
直接传给低层接口的 `[B,S,H,D]` 契约。当前 JSON
（`probes/out/thead_dense_p0_probe.json`）已标记
`probe_status=stale-layout-before-pr61-fix`，其中 max_seqlen=4、输出
误差 2.8–3.7、bias 形状结论均为错误布局产物，**不能用于 #56 判定**。
探针已改为：数学输入保留 BHSD、调用前 Q/K/V transpose 到 BSHD、
输出 transpose 回 BHSD；固定 seed，断言 max_seqlen=64，并单独比较
有效 H 区间 LSE。真机复跑命令不变：
`PPU_PROBE_DEVICE=cuda:6 python3 probes/thead_dense_p0_probe.py --item efficient_attention`。
当前环境无 PPU，本轮不以推测结果替代原始真机 JSON；复跑后按实际
输出撤回或收口 #56。

**首轮直调证据（仅作布局错误归因，不作缺口结论）**:

- op 存在: schema 与上游 v2.11.0 一致（10 位置参 + 4 kw 参,
  6 返回值: output/logsumexp/philox_seed/philox_offset/
  max_seqlen_batch_q/max_seqlen_batch_k; philox_seed 为 int64）。
- 代表用例（B=2,H=4,S=64,D=32 fp16, 无 bias, dropout=0,
  custom_mask_type=0, compute_log_sumexp=True, scale=默认 1/√D）:
  调用可执行且 6 返回值齐备，但 max_seqlen 返回 4/4（等于 H），证明
  H 被当作 S 使用；因此 output/LSE 语义无效。首轮的“output 误差”和
  “bias 均被拒”结论撤回。
- 建议开发范围（仅在正确布局复跑确认缺口后执行）: 复用该栈已验证的相邻接口
  `aten::_scaled_dot_product_efficient_attention`（ops/sdpa PLATFORM §3c
  已交付, err ≤5.62e-4）包装补齐 6 返回值契约（philox/max_seqlen 合成）,
  bias 走该接口的 `_aligned_bias` 路径; **本单先回报证据, 开发 PR 待
  范围商定后另行提交**。

## 三、issue #57: CSR SpMM 原生缺口证据

| # | 事实 | 取证 |
|---|---|---|
| 1 | `torch.sparse.mm` fp32 暖进程可用（err 8.6e-6） | 首轮组合探针 |
| 2 | fp16 路径稳定报 `cusparseSpMM_bufferSize: function is not implemented` | 同上 |
| 3 | fp32 **冷进程首发**同样报 cusparse 缺口（lazy-init 不稳定） | 隔离探针 #2 |
| 4 | 组合探针进程级 **C++ abort**（returncode -6, terminate called, 不可捕获） | 隔离探针 #3 |
| 5 | `aten::_sparse_mm` 直调 fp32 可用（err 8.6e-6） | 首轮组合探针 |

结论: 原生 CSR SpMM 路径在 PPU 栈**不可用作生产**（不稳定 + fp16 缺口 +
进程级崩溃）→ 按 issue #57「存在可复现缺口」分支自建:
**ops/spmm** = index_add 组合实现（fp32 累加, 与 Torch-FL 训练链路的
index_add 回退同构）, 注册 `aten::_sparse_mm`（torch.sparse.mm 入口;
torch.spmm=torch.mm 别名保留稠密行为不注册）, 三层测试全绿
（含 N=64/512 台账对照尺寸、空行/重复列索引/无序索引边界、反向 dB）。
Gowalla 训练端到端（Torch-FL + FlagScale 全链路）待训练环境后按
issue 验收段补 C2。

## 四、交付状态汇总

| 算子目录 | 三层测试 | 黄金 | 文档 | catalog |
|---|---|---|---|---|
| ops/gelu | ✅✅✅ | ✅（script 三件套） | ✅ | ✅ |
| ops/linear | ✅✅✅ | ✅（script 三件套） | ✅ | ✅ |
| ops/layer_norm | ✅✅✅ | 待生成 | ✅ | ✅ |
| ops/native_layer_norm | ✅✅✅ | 待生成 | ✅ | ✅ |
| ops/matmul | ✅✅✅ | 待生成 | ✅ | ✅ |
| ops/softmax | ✅✅✅ | 待生成 | ✅ | ✅ |
| ops/_softmax | ✅✅✅ | 待生成 | ✅ | ✅ |
| ops/spmm | ✅✅✅ | 待生成 | ✅ | ✅ |
| #56 _efficient_attention_forward | 证据回报（缺口确认, 开发待范围商定） | — | 本报告 | — |

C2（真实引擎 baseline/injected 双跑）: 归 thead-ppu-c2 栈
（flagrelease_ppu_vllm024_gems_540 镜像）执行, 按各算子 REPORT 的
C2 runbook 排期; 本轮 dev 栈交付为 C1 口径（framework 层 mini consumer
恒等/梯度对拍, 见各 test/framework_level.py 输出）。
