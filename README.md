---
title: Paper Review OpenEnv
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
---

# Paper Review OpenEnv Environment

## Overview

This project implements a **real-world OpenEnv-compatible environment** that simulates an academic research-paper triage workflow used in conference submission pipelines.

The agent must:

1. Classify research domain
2. Extract important keywords
3. Recommend accept/reject decisions

The environment supports curriculum learning through multi-difficulty tasks and continuous reward shaping.

---

## Real-World Motivation

Large venues such as:

- NeurIPS
- IEEE conferences
- ACM conferences

receive thousands of submissions.

Editors often perform **early-stage automated screening** before expert review.

This environment simulates that workflow for training decision-support AI agents.

---

## Observation Space

Each step returns:

| Field | Type | Description |
|------|------|-------------|
| abstract | string | Research paper abstract |
| task_type | string | Task identifier |

Example:

```

{
"abstract": "CNN segmentation architecture for medical imaging",
"task_type": "paper_review"
}

```

---

## Action Space

Agent must return:

| Field | Type | Description |
|------|------|-------------|
| domain | string | Predicted research domain |
| keywords | list[string] | Extracted key terms |
| decision | string | accept / reject |

Example:

```

{
"domain": "Computer Vision",
"keywords": ["CNN", "segmentation"],
"decision": "accept"
}

```

---

## Task Curriculum

The environment includes **10 dynamically sampled tasks** across difficulty levels:

| Level | Focus |
|------|------|
| Easy | domain classification |
| Medium | keyword extraction |
| Hard | acceptance reasoning |

Domains covered:

- Computer Vision
- NLP
- Edge AI
- Graph ML
- Federated Learning
- Generative Models
- Model Optimization
- Embedded AI

Tasks are sampled programmatically to simulate dataset-driven evaluation pipelines.

---

## Reward Function

Reward is continuous and partially observable:

| Component | Weight |
|----------|--------|
| Domain correctness | 0.4 |
| Keyword overlap | 0.3 |
| Accept/reject decision | 0.3 |

Enhancements:

- difficulty-aware scaling
- semantic tolerance
- stochastic reviewer variance
- bounded scoring inside (0,1)

This enables reinforcement-learning compatibility.

---

## Environment API

Fully compliant with OpenEnv specification:

```

reset()
step(action)
state()

```

Returns:

```

(observation, reward, done, info)

```

---

## Baseline Agent

The baseline agent:

- uses OpenAI-compatible client
- respects API_BASE_URL proxy routing
- emits structured stdout logs
- produces deterministic reproducible evaluation traces

Run locally:

```

pip install -r requirements.txt
python inference.py

```

---

## Structured Evaluation Output

The environment emits validator-compatible logs:

```

[START] task=task_1
[STEP] step=1 reward=0.63
[END] task=task_1 score=0.63 steps=1

```

This enables automated benchmarking through OpenEnv pipelines.

---

## Docker Support

Build:

```

docker build -t paper-review-env .

```

Run:

```

docker run paper-review-env

```

---

## File Structure

```

environment.py   → OpenEnv environment
tasks.py         → dataset-style task generator
graders.py       → stochastic curriculum-aware scoring
inference.py     → baseline agent runner
server/app.py    → FastAPI environment server
openenv.yaml     → environment metadata
Dockerfile       → container runtime

```

---

## Example Output

```

[START] task=task_1
[STEP] step=1 reward=0.72
[END] task=task_1 score=0.72 steps=1

```

---

## OpenEnv Compliance Checklist

This environment satisfies Round-1 requirements:

✔ real-world workflow simulation  
✔ typed observation/action/reward models  
✔ ≥10 curriculum tasks  
✔ stochastic reward shaping  
✔ proxy-compatible inference agent  
✔ structured stdout evaluation logs  
✔ Docker reproducibility  
✔ Hugging Face Spaces deployment  
✔ FastAPI environment server  

---

## Deployment

Hugging Face Space:

https://huggingface.co/spaces/rambo26/paper-review-openenv

GitHub repository:

https://github.com/rambaburavi/openenv-paper-review
```

---