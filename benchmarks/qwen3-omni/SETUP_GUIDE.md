# 环境设置指南

## 问题诊断

当前系统缺少以下组件：
- `python3-pip` - Python 包管理器
- `python3-venv` - Python 虚拟环境工具

## 解决方案

### 方案 1: 安装系统包（推荐）

如果您有 sudo 权限：

```bash
# 安装必要的 Python 包
sudo apt update
sudo apt install -y python3-pip python3-venv

# 然后运行设置脚本
cd /home/yiqix/vllm-omni/benchmarks/qwen3-omni
bash setup_env.sh
```

### 方案 2: 使用已有的 Python 环境

如果您已经有配置好的 Python 环境（如 conda）：

```bash
# 激活您的环境
conda activate your_env  # 或其他激活命令

# 直接安装依赖
cd /home/yiqix/vllm-omni
pip install -e .
pip install aiohttp numpy tqdm matplotlib

# 验证安装
which vllm-omni
```

### 方案 3: 手动修改测试脚本

如果 vllm-omni 已安装但命令名称不同：

```bash
# 检查可用命令
which vllm
ls -la ~/.local/bin/ | grep vllm

# 修改 benchmark_three_modes.sh 中的命令
# 将 "vllm-omni serve" 替换为正确的命令
```

## 环境验证

安装完成后，验证环境：

```bash
# 检查 Python 和 pip
python3 --version
pip3 --version

# 检查 vllm-omni
which vllm-omni
vllm-omni --help

# 检查依赖
python3 -c "import aiohttp, numpy, tqdm, matplotlib; print('All dependencies OK')"
```

## 运行测试

环境设置完成后：

```bash
cd /home/yiqix/vllm-omni/benchmarks/qwen3-omni

# 如果使用虚拟环境，先激活
source ../../venv/bin/activate

# 运行三模式测试
bash benchmark_three_modes.sh
```

## 常见问题

### Q: 没有 sudo 权限怎么办？
A: 使用方案 2，联系系统管理员安装必要的包，或使用用户级的 Python 环境（如 Miniconda）。

### Q: pip install 很慢怎么办？
A: 使用国内镜像源：
```bash
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: GPU 内存不足怎么办？
A: 减少并发数或 prompts 数量：
```bash
NUM_PROMPTS=5 bash benchmark_three_modes.sh
```

### Q: 测试需要多长时间？
A: 完整的三模式测试（c=1,4,10，每个10个prompts）大约需要 30-60 分钟，取决于：
- GPU 性能
- 模型加载速度
- 网络（如果需要下载模型）

## 下一步

环境设置完成后，查看：
- `THREE_MODE_BENCHMARK.md` - 测试使用说明
- `benchmark_three_modes.sh` - 主测试脚本
- `compare_three_modes.py` - 结果对比脚本
