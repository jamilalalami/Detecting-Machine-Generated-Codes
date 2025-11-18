# Detecting-Machine-Generated-Codes
 Detecting Machine-Generated Code with  Multiple Programming Languages, Generators, and Application Scenarios.

Subtask A: Binary Machine-Generated Code Detection

Goal:
Given a code snippet, predict whether it is:

(i) Fully human-written, or
(ii) Fully machine-generated
Training Languages: C++, Python, Java
Training Domain: Algorithmic (e.g., Leetcode-style problems)


# Subtask B: Multi-Class Authorship Detection
Goal:
Given a code snippet, predict its author:

(i) Human
(ii–xi) One of 10 LLM families:
DeepSeek-AI, Qwen, 01-ai, BigCode, Gemma, Phi, Meta-LLaMA, IBM-Granite, Mistral, OpenAI
Evaluation Settings:

Seen authors: Test-time generators appeared in training
Unseen authors: Test-time generators are new but from known model families



# Subtask C: Hybrid Code Detection
Goal:
Classify each code snippet as one of:

Human-written
Machine-generated
Hybrid — partially written or completed by LLM

