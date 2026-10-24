# 平台绑定清单（Platform Binding）— embedding

> 当前交付包含四个平台：**ascend910**（厂商委托：index_select 同源
> err=0 + dense_backward 委托）、**p800-kunlunxin**（厂商委托同构，
> gatherFIX 探针见 reports/）、**cambricon**（MLU590，Autograd key
> 注册——见 reports/cambricon.md §4）与 **thead-ppu**（平头哥 PPU
> ZW810E，CUDA 兼容层——见 reports/thead-ppu.md）。
> 本文件记录不可凭直觉搬运的绑定结论；跨平台移植时逐项重验，不要把
> 单平台实验结果当成普适行为。

## 1. 可直接复用（平台无关）

| 内容 | 位置 |
|---|---|
| fp32 语义参考 | `reference.py` |
| 黄金规格与数据 | `goldendata/`（CPU 生成，天然跨平台） |
| 平台 facade（按张量设备路由） | `kernel/platform.py` |
| 三层测试、性能脚本 | `test/`、`script/`（`EMBEDDING_PROFILE` 切平台） |
| A1 autograd.Function 包装 | `register.py`（`_BACKENDS` 选择器 + `_DEVICE_PROBE` 守卫） |

## 2. ascend910 绑定（厂商委托）

| 绑定项 | 实测结论 |
|---|---|
| A1 拦截 key | **AutogradPrivateUse1**（PrivateUse1 永不命中——sdpa 开发报告 §3.1 dispatch 证据链同结论） |
| aten 签名坑 | `aten::embedding_dense_backward` **5 参无 sparse**（与 Python 层 6 参不同） |
| autograd.Function | staticmethod 读模块全局 → `register_a1` 里 `global` 全部三变量（PLATFORM/embedding/embedding_backward） |
| 同步 API | `torch.npu.synchronize`（非 cuda）；`kernel/platform.py::synchronize` 已平台化 |
| 性能 | 委托=包装开销 0.62-0.99x native；Triton gather 探针 18-69x 慢（P800 结论在 Ascend 复现——embedding 是带宽型，自研 Triton 无优势） |

## 3. p800-kunlunxin 绑定

- XMLIR 将 XPU 呈现为 **CUDA 张量**（device.type=="cuda"），dispatch
  key 为 AutogradCUDA
- gatherFIX 探针结论见 `reports/`（正确性 0 error；特化比 native 慢
  10.9-177.9x——与 ascend910 的 Triton 探针结论一致）

## 4. cambricon（MLU590）绑定

- **Autograd key** 注册（与 configs/devices/cambricon.yaml 的
  PrivateUse1 不同）：embedding 带 autograd 的 A1 交付须注册 Autograd
  key 才能携带自带 backward；两个 key 实测都能命中，取 Autograd key
  以免覆盖 functorch 的 PrivateUse1 batch rule（reports/cambricon.md §4）
- 与 ascend910 同为 AutogradPrivateUse1 时：单机双 NPU+MLU 场景须
  `EMBEDDING_PROFILE` 显式指定平台（`_backend_for` 的 platform 参数
  优先于 dispatch_key 推断）

## 5. thead-ppu（PPU ZW810E）绑定

- PPU torch 经 **CUDA 兼容层**直曝 `cuda:N` 张量——与 XMLIR 伪装 CUDA
  同 device.type，`kernel/platform.py::_cuda_impl` 按栈身份路由：
  `torch_xmlir` 可导入 → p800；设备名含 `PPU` → thead；进程级缓存
  （T1 教训: 无缓存 find_spec 路由 80-100µs/调用）。`register.py`
  `_cuda_backend` 同款。
- A1 拦截 key = **AutogradCUDA**（probes/thead_first_probe.py 实测
  6/7 命中; 唯一零命中 inference_mode = known-issues #19 Autograd 键
  通病, 与 p800/T1-sdpa 同结论）。
- **原生 `embedding_backward` 完整**（`padding_idx` 与
  `scale_grad_by_freq=True` 直通, fp16 err 6.1e-5~1.2e-4 累加序噪声）
  ——与 cambricon 同型零补偿；若未来栈升级出现 p800 式
  `Check scale_grad_by_freq == false failed`，回退 p800 的
  inverse-frequency 补偿（p800_kunlunxin.py 已有）。
- forward native `embedding`/`index_select` vs CPU 官方**逐位 err=0**
  （3 dtype × padding 全组合）。
- 同步 API：`torch.cuda.synchronize`（CUDA 兼容层提供），
  `kernel/platform.py::synchronize` cuda 分支直接覆盖。
- 测试尾块维口径：§2d 的 3072/5000/5120 对 embedding 取 gather 长度维
  （#15a 尾块 masked-load 归约陷阱不适用——gather 无归约）。
- dev 栈无 flag_gems：framework_level 跳过 FlagOS 栈门禁（纯 A1 拦截
  口径）；`run.py --consistency`/Triton gather 对比结构性缺证（同 T1
  sdpa 报告 §6.11）。

## 6. 新平台接入

按 `MERGE.md` 七步走；kernel 层实现 `embedding`/`embedding_backward`
+ `PLATFORM` + `_DEVICE_PROBE` 四件即可并入 `_BACKENDS`。
