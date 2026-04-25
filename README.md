# telegram-bot-on-llm

Telegram bot running on k3s, backed by a local Ollama LLM on kbrain.

git push -> Github Action -> Docker Hub -> ArgoCD -> k3s

## How it works

The bot uses long-polling - no webhook, no ingress needed. On startup it auto-detects the first available Ollama model. Messages are processed sequentially (Ollama handles one request at a time).

## Stack

`k3s` · `ArgoCD` · `Helm` · `GitHub Actions` · `Docker Hub` · `FastAPI` · `uv` · `Ollama`

## Bootstrap

```bash
# 1. create the token secret before ArgoCD syncs
kubectl create secret generic telegram-bot-on-llm-secret \
  --from-literal=TELEGRAM_TOKEN=<your-token> \
  -n telegram-bot-on-llm

# 2. register the app with ArgoCD
kubectl apply -f https://raw.githubusercontent.com/DmytroKrynytsyn/telegram-bot-on-llm/main/argocd/application.yaml
```