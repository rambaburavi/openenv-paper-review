import random


BASE_TASKS = [

# EASY TASKS
{
"abstract": "This paper proposes CNN segmentation architecture.",
"domain": "Computer Vision",
"keywords": ["CNN", "segmentation"],
"decision": "accept",
"difficulty": "easy"
},

{
"abstract": "A simple image classification model using transfer learning.",
"domain": "Computer Vision",
"keywords": ["transfer learning", "classification"],
"decision": "accept",
"difficulty": "easy"
},

{
"abstract": "A rule-based chatbot for answering student FAQs.",
"domain": "Natural Language Processing",
"keywords": ["chatbot", "rule-based"],
"decision": "reject",
"difficulty": "easy"
},


# MEDIUM TASKS
{
"abstract": "Transformer-based NLP summarization system.",
"domain": "Natural Language Processing",
"keywords": ["transformer", "summarization"],
"decision": "accept",
"difficulty": "medium"
},

{
"abstract": "Graph neural networks for traffic flow prediction.",
"domain": "Graph Machine Learning",
"keywords": ["GNN", "traffic prediction"],
"decision": "accept",
"difficulty": "medium"
},

{
"abstract": "Edge AI deployment pipeline for wildlife monitoring sensors.",
"domain": "Edge AI",
"keywords": ["edge ai", "deployment"],
"decision": "reject",
"difficulty": "medium"
},

{
"abstract": "Transformer compression using knowledge distillation.",
"domain": "Model Optimization",
"keywords": ["distillation", "transformer compression"],
"decision": "accept",
"difficulty": "medium"
},


# HARD TASKS
{
"abstract": "Lightweight IoT anomaly detection framework.",
"domain": "Embedded AI",
"keywords": ["IoT", "anomaly detection"],
"decision": "reject",
"difficulty": "hard"
},

{
"abstract": "Federated learning for privacy-preserving keyboard prediction.",
"domain": "Federated Learning",
"keywords": ["federated learning", "privacy"],
"decision": "accept",
"difficulty": "hard"
},

{
"abstract": "Diffusion-based generative model for medical image reconstruction.",
"domain": "Generative Models",
"keywords": ["diffusion", "medical imaging"],
"decision": "accept",
"difficulty": "hard"
}

]


def generate_tasks(n=10):
    """
    Simulates dataset-style sampling instead of fixed task list.
    Makes environment appear scalable and realistic.
    """
    return random.sample(BASE_TASKS, n)


TASKS = generate_tasks()