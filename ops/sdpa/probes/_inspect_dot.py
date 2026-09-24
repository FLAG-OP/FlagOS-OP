import inspect

import triton.language as tl

print("tl.dot signature:", inspect.signature(tl.dot))
src = inspect.getsource(tl.dot)
print(src[:1200])
