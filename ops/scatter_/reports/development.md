# 开发记录

1. 需求：ATen 原地 scatter 的 Tensor / Scalar 无归约重载。
2. 参考：仓库 operator 模板、copy_ 实现、Dashener2 PR #42。
3. 算法：按 index 元素展开、独立 stride 地址、索引维替换。
4. 正确性：精确复制，int64 地址，独立标量尾部，写入前范围检查。
5. 注册：CUDA backend，保留原生 autograd/in-place 分发层。
6. 测试：CPU 元数据；设备直测、拦截计数、mini consumer。
7. 实测：用户授权 0 号卡后完成 P800 gems5 三层、黄金与性能；速度低于原生。

XMLIR 兼容性问题与处理：

- constexpr tuple 不能直接下标访问，使用 `.value[d]`。
- masked scatter 尾部曾污染目标首元素；将尾部重复映射到最后一个有效元素
  仍出现偶发错值（float32 5×7×13）。改为独立 BLOCK=1 尾部程序，
  kernel/A1 10 轮回归通过。
- device_assert 未报错；增加原生布尔范围检查 + any().item()，在写入前拒绝。
- 原生 int64 aminmax 不支持，范围检查使用比较与 any，不做有损 dtype 转换。
