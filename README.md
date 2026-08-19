# Qwen3.8-27B-Uncensored-MLX 本地部署调查与操作手册

本文记录 `orcarouter/Qwen3.8-27B-Uncensored-MLX` 在当前 Mac Studio 上的可运行性调查、实际部署步骤、测试结果和已知限制。记录日期：2026-08-19（Asia/Tokyo）。

## 1. 结论

- 当前机器可以轻松运行该模型的最高质量版本 `8-bit`，并已实际加载、生成文本和通过 OpenAI 兼容 API 验证。
- 本仓库最高只提供 8-bit 量化，因此这里的“满血”是指仓库内最高精度，并非 BF16 原始精度。
- 模型的原生、已验证配置上下文为 **262,144 tokens**。官方基础模型说明可以通过 YaRN 外推至 1,000,000 tokens，但本次 MLX 部署没有验证 1M，生产使用应把 262,144 当作可靠上限。
- 当前服务运行在 `http://127.0.0.1:8080`，仅监听本机。
- 实测短上下文生成速度约为 **21–25 tokens/s**，峰值内存约 **34GB**。
- 配置文件声明模型带一层 MTP，但下载到的权重中没有 `mtp.*` 张量，因此无法启用原生 MTP 推测解码。普通推理、视觉输入和模型质量不受此问题影响。

## 2. 当前硬件与软件

### 硬件

| 项目 | 配置 |
| --- | --- |
| 机型 | Mac Studio `Mac15,14` |
| 芯片 | Apple M3 Ultra |
| CPU | 32 核：24 性能核 + 8 能效核 |
| GPU | 80 核 |
| 统一内存 | 512GB |
| 系统 | macOS 26.5.2，Build 25F84 |
| 部署后可用磁盘 | 约 126GiB |

### 软件

| 软件 | 版本 |
| --- | --- |
| Python | 3.14.6 |
| MLX | 0.32.1 |
| mlx-vlm | 0.6.15 |
| huggingface-hub | 1.28.0 |
| Jinja2 | 3.1.6 |

Python 环境位于仓库内：

```text
.venv/
```

模型位于仓库内（不纳入 Git）：

```text
models/Qwen3.8-27B-Uncensored-MLX/8-bit/
```

## 3. 模型调查结果

该模型是 Qwen3.8-27B 的 abliterated（移除拒答方向）版本，使用 MLX affine 量化，group size 为 64。视觉塔、归一化层和部分卷积层保留 BF16，语言模型线性权重被量化。

仓库提供以下版本：

| 子目录 | 权重大小 | 说明 |
| --- | ---: | --- |
| `2-bit/` | 约 8.69GiB | 质量严重下降，不建议实际使用 |
| `4-bit/` | 约 14.95GiB | 默认版本，速度和质量平衡 |
| `6-bit/` | 约 21.21GiB | 较高质量 |
| `8-bit/` | 约 27.48GiB | 仓库内最高质量，本次部署版本 |

仓库根目录还复制了一份 4-bit 权重。若直接下载整个仓库，会同时下载全部量化版本和重复的根目录 4-bit，因此应使用 `--include "8-bit/*"` 只取需要的版本。

关键模型配置：

| 配置 | 值 |
| --- | ---: |
| 架构 | `Qwen3_5ForConditionalGeneration` |
| 总层数 | 64 |
| 隐藏维度 | 5120 |
| 全注意力间隔 | 每 4 层一次 |
| 全注意力层 | 16 |
| 线性注意力层 | 48 |
| KV heads | 4 |
| Head dimension | 256 |
| 原生上下文 | 262,144 tokens |

按 BF16 KV cache 粗略估算，16 个全注意力层在 262,144 tokens 时需要约 16GiB 的增长型 KV cache：

```text
16 layers × 2(K/V) × 4 KV heads × 256 dim × 2 bytes × 262,144 tokens
≈ 16GiB
```

再加约 27.5GiB 权重和运行时开销，仍远低于 512GB 统一内存。容量不是瓶颈；超长 prompt 的预填充时间和长上下文解码速度才是主要限制。

## 4. 从零部署

以下命令都在项目目录执行：

```bash
cd qwen3.8
```

### 4.1 接受仓库条款并登录 Hugging Face

该仓库为 gated repository。先在浏览器打开模型页面、接受条款并允许共享联系信息：

<https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-MLX>

然后登录 CLI：

```bash
hf auth login
hf auth whoami
```

### 4.2 创建独立 Python 环境

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
python -m pip install -U \
  "mlx>=0.32" \
  "mlx-vlm>=0.6.13" \
  huggingface_hub \
  jinja2
```

`jinja2` 需要显式安装。本次首次运行时，`mlx-vlm` 已安装但缺少该运行时依赖，导致聊天模板编译失败。

### 4.3 只下载 8-bit 版本

```bash
mkdir -p models

hf download orcarouter/Qwen3.8-27B-Uncensored-MLX \
  --include "8-bit/*" \
  --local-dir ./models/Qwen3.8-27B-Uncensored-MLX
```

下载完成后应存在 6 个权重分片：

```bash
find models/Qwen3.8-27B-Uncensored-MLX/8-bit \
  -maxdepth 1 \
  -name 'model-*.safetensors' \
  -print | sort
```

检查量化和上下文配置：

```bash
jq '{
  model_type,
  architectures,
  quantization,
  max_position_embeddings: .text_config.max_position_embeddings,
  num_hidden_layers: .text_config.num_hidden_layers,
  full_attention_interval: .text_config.full_attention_interval
}' models/Qwen3.8-27B-Uncensored-MLX/8-bit/config.json
```

预期关键结果：

```json
{
  "model_type": "qwen3_5",
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "quantization": {
    "bits": 8,
    "group_size": 64,
    "mode": "affine"
  },
  "max_position_embeddings": 262144,
  "num_hidden_layers": 64,
  "full_attention_interval": 4
}
```

## 5. 本地推理测试

运行一次关闭思考模式的短测试：

```bash
source .venv/bin/activate

python -m mlx_vlm generate \
  --model ./models/Qwen3.8-27B-Uncensored-MLX/8-bit \
  --prompt '只回答一句话：本地模型启动成功。' \
  --max-tokens 64 \
  --temperature 0 \
  --thinking-mode disabled \
  --verbose
```

本次实测结果：

```text
输出：本地模型启动成功。
Prompt：21 tokens，10.492 tokens/s
Generation：6 tokens，25.453 tokens/s
Peak memory：34.028GB
```

这是很短的单次测试，只适合作为加载成功和速度量级参考。实际速度会随上下文长度、输出长度、思考模式、温度和并发数变化。

## 6. 启动 OpenAI 兼容服务

```bash
cd qwen3.8
source .venv/bin/activate

python -m mlx_vlm server \
  --model ./models/Qwen3.8-27B-Uncensored-MLX/8-bit \
  --host 127.0.0.1 \
  --port 8080
```

保持此终端窗口运行。服务地址：

```text
http://127.0.0.1:8080
```

OpenAI 客户端的 Base URL：

```text
http://127.0.0.1:8080/v1
```

默认没有配置 API key。部分客户端强制要求填写时，可填任意非空占位值，但服务端不会校验它。

### 6.1 健康检查

```bash
curl --silent http://127.0.0.1:8080/health | jq .
```

本次健康检查确认：

```json
{
  "status": "healthy",
  "loaded_model": "./models/Qwen3.8-27B-Uncensored-MLX/8-bit",
  "effective_context_limit": 262144,
  "continuous_batching_enabled": true,
  "apc_enabled": false
}
```

### 6.2 查看模型 ID

```bash
curl --silent http://127.0.0.1:8080/v1/models | jq .
```

请求中的 `model` 必须与服务加载的模型 ID 完全一致：

```text
./models/Qwen3.8-27B-Uncensored-MLX/8-bit
```

不要随意填 `local-qwen3.8-27b` 等别名。该服务器支持动态模型切换；如果收到不同的模型 ID，它会卸载当前模型并尝试从本地路径或 Hugging Face 加载新 ID。

### 6.3 Chat Completions 调用

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "./models/Qwen3.8-27B-Uncensored-MLX/8-bit",
    "messages": [
      {"role": "user", "content": "你好，请简单介绍一下自己"}
    ],
    "temperature": 0.7,
    "max_tokens": 512,
    "enable_thinking": false
  }'
```

启用思考模式：

```json
"enable_thinking": true
```

API 端到端测试结果：

```text
输入：只回答：API 正常
输出：API 正常
Prompt：18 tokens，59.92 tokens/s
Generation：4 tokens，21.28 tokens/s
Peak memory：34.01GB
```

## 7. 停止和重启

前台运行时，直接在服务终端按 `Ctrl+C`。

也可以先查找监听 8080 端口的 PID：

```bash
lsof -nP -iTCP:8080 -sTCP:LISTEN
```

然后停止对应进程：

```bash
kill <PID>
```

重新启动时重复第 6 节命令即可，不需要重新下载模型。

## 8. 已知问题与排错

### 8.1 下载返回 401 或无权访问

原因通常是尚未在网页接受 gated repository 条款，或 CLI 没有登录。

```bash
hf auth whoami
```

若未登录：

```bash
hf auth login
```

### 8.2 `apply_chat_template requires jinja2`

安装缺少的依赖：

```bash
source .venv/bin/activate
python -m pip install -U jinja2
```

### 8.3 请求错误模型 ID 后服务尝试联网下载

API 请求的 `model` 应使用：

```text
./models/Qwen3.8-27B-Uncensored-MLX/8-bit
```

若误填其他 ID，重新用正确 ID 发请求即可；服务器会重新加载本地 8-bit 模型。用 `/health` 确认最终状态。

### 8.4 MTP 无法启用

模型配置包含：

```json
"mtp_num_hidden_layers": 1
```

但 `model.safetensors.index.json` 中没有任何 `mtp.*`、`nextn.*` 或额外 `layers.64.*` 权重。运行 mlx-vlm 的 Qwen MTP 分离器会报错：

```text
ValueError: No mtp.* tensors found
```

因此当前仓库不能使用原生 MTP 推测解码。这只影响潜在的解码加速，不影响正常输出。不要仅根据配置字段判断 MTP 权重实际存在。

### 8.5 262K 能装下，但不代表交互速度不变

KV cache 和注意力计算会随上下文增长。机器内存足够装下完整 262K 窗口，但长 prompt 的首 token 等待时间可能达到分钟级，后续生成速度也会下降。建议按实际任务逐级测试 32K、64K、128K，再使用完整 262K。

### 8.6 1M 上下文

官方基础模型称可通过 YaRN 外推到 1M，但这不是模型原生窗口。本次没有修改 MLX 配置、没有验证 MLX 下的 YaRN 质量，也没有进行 1M 压力测试。因此当前部署的支持边界仍是 262,144 tokens。

### 8.7 KV cache 量化与 APC

本次保持默认配置：

- KV cache 未量化；512GB 内存下没有必要为节省容量承担额外兼容性和性能风险。
- Continuous batching 已启用。
- APC 自动前缀缓存未启用。

如需面向长前缀、多轮代理或并发服务进一步优化，应单独做正确性、命中率、吞吐和内存对照测试，避免直接修改生产参数。

## 9. 安全注意事项

该模型经过 refusal removal，缺少可靠的内置安全护栏。当前使用 `127.0.0.1` 只对本机开放是有意设置。

不要直接把无鉴权的服务绑定到公网或不可信局域网。若确实需要远程访问，应至少配置服务端 API key、防火墙、访问控制和独立内容安全层。

## 10. 参考资料

- 目标模型：<https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-MLX>
- 官方基础模型：<https://huggingface.co/Qwen/Qwen3.8-27B>
- 官方模型配置：<https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json>
- mlx-vlm：<https://github.com/Blaizzy/mlx-vlm>
- mlx-lm：<https://github.com/ml-explore/mlx-lm>
- M3 Ultra 长上下文参考测试：<https://github.com/ml-explore/mlx/discussions/3209>
