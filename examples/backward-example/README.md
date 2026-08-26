# backward-example: 自定义算子反向传播

[← 样例索引](../README.md)

## 定位

补齐训练侧——此前样例全部为 forward-only。本样例演示:

1. `torch.autograd.Function` 包装自定义 kernel（forward + backward 均为 Triton）
2. backward kernel 推导 gelu_and_mul 的解析梯度
3. 与 PyTorch autograd 数值梯度对比验证
4. 训练冒烟（SGD 一步）

## 运行

```bash
python3 examples/backward-example/example.py
```

## 实测

```
forward:  PASS
backward: dx_err=1.7e-06 dg_err=5.7e-07 PASS（vs PyTorch autograd）
训练冒烟: PASS
```

## 数学

y = gelu(x) * g, 其中 gelu 为 tanh 近似:

```
dy/dx = gelu'(x) * g
dy/dg = gelu(x)
gelu'(x) = 0.5*(1+tanh(inner)) + x*0.5*sech²(inner)*inner'
inner = sqrt(2/π)*(x + 0.044715x³)
inner' = sqrt(2/π)*(1 + 3*0.044715x²)
```

## autograd.Function 注册模式

```python
class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate):
        out = triton_forward(x, gate)
        ctx.save_for_backward(x, gate)
        ctx.input_shapes = (x.shape, gate.shape)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, gate = ctx.saved_tensors
        dx, dgate = triton_backward(x, gate, grad_out)
        return dx.reshape(ctx.input_shapes[0]), dgate.reshape(ctx.input_shapes[1])
```

> 注意: `ctx.save_for_backward` 保存 flatten 后的张量，backward 中
> 用 `ctx.input_shapes` 恢复形状。
