# scatter_ · Ascend910 交付记录（含 kernelGen 使用）

## kernelGen 使用
- 用户提供了 kernelGen 生成的 `scatter_.md`（TLE/DSA 版本）作为初稿。
- 在本机（Ascend910 / triton 3.5.1+flagtree）实测：TLE 版本的串行 `tle.dsa.copy`
  在 `TritonToLinalgIncubated` 阶段编译失败（41 例中 40 失败）。
- 期间 kernelGen MCP 后端多次返回“无法连接到后端服务/504”，无法在线迭代。
- 处理：参考该初稿的语义（scatter_.src/value、负 dim、非连续、partial/empty index、
  重复索引、bool/int 等），**改写为纯 Triton**（按坐标计算 dst/src 偏移，
  不使用 TLE），在 Ascend 上验证全绿。

## Ascend 适配要点
- A1：注册 `scatter_.src` / `scatter_.value` 到 `AutogradPrivateUse1`（+PrivateUse1）。
- BLOCK=1024（int64 索引/偏移，4096 触发 UB 溢出）。
- Python 侧越界检查；`_version` 由实现递增（Autograd key 下 dispatcher 不自增）。
- fp64 回退原生 redispatch（BiSheng 不支持 64-bit float）。

## 结果（Ascend910，npu:0）
- kernel 层：41 + 边界 11 全过（精确相等）。
- op 层：41 次命中，原地返回、`_version+1` 正确。
- 黄金：41/41（SHA256 校验 + 精确回放）。
- framework：mini consumer（token-selection）输出一致，命中 1。
- 真实 vLLM C2：PlatformFL/npu 单进程；top-k/top-p 采样命中 scatter_.src 8 次，输出与基线逐 token 一致。
- 级别：C2（src，complete-pending-review）；value 为 C1。
