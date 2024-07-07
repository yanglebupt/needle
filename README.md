# Needle
My implementation of Needle：CMU's Deep Learning Systems

[Build your own deep learning framework based on C/C++, CUDA, and Python, with GPU support and automatic differentiation](https://towardsdatascience.com/recreating-pytorch-from-scratch-with-gpu-support-and-automatic-differentiation-8f565122a3cc)

[学习文档](https://qwuzvjx4mo.feishu.cn/drive/folder/EkWOf3rasluPhEd69vscVdnPnPq?from=from_copylink)

## Development

配置环境添加 `export PYTHONPATH="/root/code/needle:$PYTHONPATH"`

```bash
vim ~/.bashrc
source ~/.bashrc
```

在 `ipynb` 中添加

```python
import sys
sys.path.append("/root/code/needle")
```

同时需要切换路径，这样才能只执行 needle 下面的 tests，而不受外部干扰

```
%cd /root/code/needle
```