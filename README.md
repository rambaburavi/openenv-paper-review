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

This project implements a real-world OpenEnv-compatible environment that simulates an AI research paper screening workflow used in academic conference submission pipelines.

The environment allows an agent to:

1. Classify the domain of a research paper
2. Extract important keywords
3. Recommend acceptance or rejection decisions

It follows the OpenEnv API specification:

* reset()
* step(action)
* state()

and supports deterministic scoring using task-specific graders.

---

## Real-World Motivation

This environment simulates early-stage reviewer triage workflows used in major research venues such as:

* NeurIPS
* IEEE conferences
* ACM conferences

AI agents assist editors by classifying submissions, extracting metadata, and recommending acceptance decisions before expert review.

---

## Observation Space

The observation returned to the agent contains:

| Field     | Type   | Description             |
| --------- | ------ | ----------------------- |
| abstract  | string | Research paper abstract |
| task_type | string | Current task stage      |
Example:
{
"abstract": "This paper proposes CNN segmentation architecture.",
"task_type": "paper_review"
}
---
## Action Space
The agent must respond with:
| Field    | Type         | Description                  |
| -------- | ------------ | ---------------------------- |
| domain   | string       | Predicted research domain    |
| keywords | list[string] | Extracted important keywords |
| decision | string       | accept or reject             |
Example:
{
"domain": "Computer Vision",
"keywords": ["CNN", "segmentation"],
"decision": "accept"
}
---
## Reward Function
The environment provides partial reward signals:
| Component                     | Score |
| ----------------------------- | ----- |
| Correct domain classification | +0.4  |
| Keyword overlap accuracy      | +0.3  |
| Correct acceptance decision   | +0.3  |
Total reward range:
0.0 → 1.0
This ensures smooth learning signals across the episode.
---
## Tasks
The environment currently includes three tasks:
### Easy
Domain classification for Computer Vision paper
### Medium
Keyword extraction for NLP summarization paper
### Hard
Acceptance decision for Embedded AI anomaly detection paper
Each task is evaluated using deterministic graders returning scores between:
0.0 and 1.0
---
## Environment API
Supported OpenEnv interface:
reset()
Returns initial observation
step(action)
Returns:
(observation, reward, done, info)
state()
Returns internal environment task state
---
## Baseline Inference Script
The baseline agent:
* Uses OpenAI client when API key available
* Falls back to deterministic heuristic predictions otherwise
* Produces reproducible scores across all tasks
Run locally:
pip install -r requirements.txt
python inference.py
---
## Docker Support
This environment is containerized and runs with:
docker build -t paper-review-env .
docker run paper-review-env
---
## File Structure
environment.py → OpenEnv environment implementation
tasks.py → task dataset definitions
graders.py → scoring logic
inference.py → baseline evaluation script
openenv.yaml → OpenEnv metadata specification
Dockerfile → container runtime configuration
---
## Expected Output Example
Task 1 Score: 1.0
Task 2 Score: 1.0
Task 3 Score: 1.0
Average Score: 1.0
---
## Compliance Checklist
This environment satisfies OpenEnv Round-1 requirements:
Real-world task simulation
Typed observation/action/reward models
3 graded tasks (easy → medium → hard)
Deterministic reward function
Baseline inference script
Docker container support
Hugging Face Spaces deployment
