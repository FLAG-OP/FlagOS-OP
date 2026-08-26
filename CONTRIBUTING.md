# 贡献指南

## 快速开始

```bash
git clone git@github.com:TruNcat3/FlagOS-OP.git
cd FlagOS-OP
python3 run.py --list                    # 查看设备与矩阵
python3 run.py --route a1 --level op     # 跑最轻的一格
```

## 贡献类型与检查清单

### 新增样例

- [ ] 复制最接近的现有样例目录为起点
- [ ] `example.py` 含 `main()` 入口 + 设备 profile 参数（不硬编码设备）
- [ ] README 含: 定位 / 运行方式 / 预期输出 / 关键点
- [ ] 若含新 kernel: 确认启动包了 `torch_device_fn.device` 上下文（[known-issues #11](docs/known-issues.md)）
- [ ] 标注[开发层级](docs/architecture.md#levels)（TR/FW/HW）
- [ ] 更新 `examples/README.md` 索引表与定位图

### 新增设备 profile

- [ ] 复制 `configs/devices/_template.yaml`
- [ ] 填写 `vendor`（与 flag_gems `DeviceDetector().vendor_name` 一致）
- [ ] 声明 `vendor_kernels`（含哨兵负例）
- [ ] 生成[黄金](docs/testing.md#golden) + 跑[漂移实验](docs/testing.md#drift)建立基线
- [ ] 用[哨兵检查](docs/testing.md#sentinel)建立该设备的已知问题清单

### 修改核心测试

- [ ] 保持 `run(profile) -> bool` 签名
- [ ] 不引入硬编码设备串/卡号/厂商库名
- [ ] 修改后跑通该路线全部三层

## 提交前检查

```bash
# 文档链接与 mermaid 完整性（0 错误才过）
# Python 语法（0 错误才过）
# 相关矩阵格全绿
```

## 提交规范

```
<类型>: <描述>

类型: feat / fix / docs / audit / refactor / test
```

## 已知重要约束

1. **不要设 `VLLM_FL_PREFER=flagos`**（参考实例 P800）——见
   [known-issues #1](docs/known-issues.md)
2. 裸 Triton 启动必须包 `torch_device_fn.device` 上下文——见
   [known-issues #11](docs/known-issues.md)
3. 插件入口必须是 `register` / `vllm_fl_register`——见
   [known-issues #8](docs/known-issues.md)

## 路线图（欢迎贡献）

| 方向 | 说明 | 参考 |
---|---|---|
| ~~reduction 类算子样例~~ | ✅ softmax-fullstack 已完成 | — |
| autotune 集成 | `@triton.autotune` / FlagGems `libtuner` 分块参数搜索 | FlagGems `bmm.py` |
| backward / autograd | 自定义算子的反向传播注册 | FlagGems `silu_and_mul_grad` |
| 多芯片 CI | 矩阵在多设备上的自动回归 | GitHub Actions self-hosted runner |
