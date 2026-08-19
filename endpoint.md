# Qwen3.8 本地服务连接信息

## OpenAI 兼容配置

| 配置项 | 值 |
| --- | --- |
| Base URL | `http://127.0.0.1:8080/v1` |
| Chat Completions | `http://127.0.0.1:8080/v1/chat/completions` |
| Models | `http://127.0.0.1:8080/v1/models` |
| Health | `http://127.0.0.1:8080/health` |
| Model ID | `./models/Qwen3.8-27B-Uncensored-MLX/8-bit` |
| API Key | 未启用；客户端强制要求时填写任意非空占位值，例如 `local` |
| 上下文上限 | `262144` tokens |

服务只监听 `127.0.0.1`，因此只能从当前 Mac 访问。

## 环境变量

```bash
export OPENAI_BASE_URL='http://127.0.0.1:8080/v1'
export OPENAI_API_KEY='local'
export OPENAI_MODEL='./models/Qwen3.8-27B-Uncensored-MLX/8-bit'
```

## 请求示例

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

启用思考模式时，将 `enable_thinking` 改为 `true`。

## 健康检查

```bash
curl http://127.0.0.1:8080/health
```

当前已验证状态：`healthy`。

## Pi Coding Agent

Pi 已在 `~/.pi/agent/models.json` 中配置 `mlx-local` provider，并在 `~/.pi/agent/settings.json` 中设为默认：

```text
Provider: mlx-local
Model: ./models/Qwen3.8-27B-Uncensored-MLX/8-bit
Thinking: supported
Images: supported
Context: 262144
Max output: 32768
```

直接启动 Pi 即可使用本地模型：

```bash
cd qwen3.8
pi
```

检查模型是否已注册：

```bash
pi --list-models mlx-local
```

非交互测试：

```bash
pi --thinking off --no-tools --no-session -p '只回答 PI_OK'
```

原有 `ds4` provider 仍保留，可在 Pi 的 `/model` 选择器中切换。

## 自动启动

MLX 服务由以下 LaunchAgent 管理：

```text
~/Library/LaunchAgents/ai.orcarouter.qwen38-mlx.plist
```

它会在用户登录后自动启动，并在异常退出时重启。查看状态：

```bash
launchctl print gui/$(id -u)/ai.orcarouter.qwen38-mlx
```

重启服务：

```bash
launchctl kickstart -k gui/$(id -u)/ai.orcarouter.qwen38-mlx
```

## 注意事项

请求中的 `model` 必须与上面的 Model ID 完全一致。该服务支持动态加载模型；填写其他名称可能导致当前模型被卸载，并触发本地查找或 Hugging Face 下载。
