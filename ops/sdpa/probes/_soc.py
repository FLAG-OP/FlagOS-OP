import torch
import torch_npu

v = torch_npu.npu.get_soc_version()
# torch_npu SocVersion 枚举: 253 = Ascend910B ? 打表确认
try:
    from torch_npu._C import SocVersion
    names = [n for n in dir(SocVersion) if not n.startswith("_")]
    for n in names:
        if getattr(SocVersion, n) == v:
            print("soc:", n, "=", v)
            break
    else:
        print("soc enum value:", v, "| all:", names[:20])
except Exception as e:
    print("enum probe fail:", e, "| raw:", v)
print("device name:", torch.npu.get_device_properties(0).name
      if hasattr(torch.npu.get_device_properties(0), "name") else "?")
