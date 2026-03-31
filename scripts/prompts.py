"""Diverse prompt corpus for cache benchmarking.

Covers domains that activate different expert subsets:
- English technical (code, science)
- Mathematics (formulas, proofs)
- Creative writing (fiction, poetry)
- Multilingual (Russian, Chinese, Arabic)
- Conversational (chat, Q&A)
- Domain shifts (switching between the above)

Based on methodology from HOBBIT (arxiv 2411.01433), ExpertFlow (arxiv 2410.17954),
and the Routing Consistency Study (arxiv 2505.16056).
"""

# Phase 1: Cold start — diverse first prompts (no cache warmth)
COLD_START = [
    "Write a Python function that implements quicksort with type hints.",
    "Explain the proof of Fermat's Last Theorem in accessible language.",
    "The rain drummed against the windowpane as Sarah opened the letter.",
    "Опишите принцип работы квантового компьютера простым языком.",
    "请解释量子纠缠的基本原理。",
    "What are the main causes and effects of ocean acidification?",
    "Derive the Euler-Lagrange equation from the principle of least action.",
    "Write a haiku about a programmer debugging at 3am.",
]

# Phase 2: Domain-specific sustained generation
CODE_PROMPTS = [
    "Implement a red-black tree in Rust with insert, delete, and rebalance.",
    "Write a CUDA kernel for matrix multiplication with shared memory tiling.",
    "Design a lock-free concurrent hash map in C++ using atomic operations.",
    "Implement the Raft consensus algorithm in Go with leader election.",
]

MATH_PROMPTS = [
    "Prove that the sum of 1/n^2 from n=1 to infinity equals pi^2/6.",
    "Solve the heat equation on a finite rod with Dirichlet boundary conditions.",
    "Derive the Black-Scholes formula for European call option pricing.",
    "Prove the Banach fixed-point theorem and give three applications.",
]

CREATIVE_PROMPTS = [
    "Write the opening chapter of a noir detective novel set in 1940s Shanghai.",
    "Compose a Shakespearean sonnet about artificial intelligence.",
    "Write a short story about a lighthouse keeper who discovers time travel.",
    "Create a dialogue between Socrates and a modern AI researcher.",
]

MULTILINGUAL_PROMPTS = [
    "Напишите эссе о влиянии Достоевского на мировую литературу.",
    "用中文详细解释深度学习中的注意力机制。",
    "اشرح نظرية النسبية العامة لأينشتاين بالعربية",
    "Erklären Sie die Grundprinzipien der Quantenmechanik auf Deutsch.",
]

CONVERSATION_PROMPTS = [
    "I just got a puppy and it keeps chewing my shoes. What should I do?",
    "Can you help me plan a week-long trip to Japan on a budget?",
    "My code compiles but gives wrong output. Here's the function: def fib(n)...",
    "What's the difference between a latte, cappuccino, and flat white?",
]

# Phase 3: Domain shift sequences — each tuple is (warmup_domain, shift_domain)
DOMAIN_SHIFTS = [
    ("code", "creative"),       # technical → creative
    ("math", "conversation"),   # formal → casual
    ("code", "multilingual"),   # English → non-English
    ("creative", "math"),       # narrative → symbolic
    ("multilingual", "code"),   # non-English → English technical
]

DOMAIN_MAP = {
    "code": CODE_PROMPTS,
    "math": MATH_PROMPTS,
    "creative": CREATIVE_PROMPTS,
    "multilingual": MULTILINGUAL_PROMPTS,
    "conversation": CONVERSATION_PROMPTS,
}

# Phase 4: ShareGPT-style multi-turn (simulated)
MULTI_TURN = [
    [
        "What is a neural network?",
        "How does backpropagation work?",
        "Can you show me a simple implementation in PyTorch?",
        "Now modify it to use batch normalization.",
    ],
    [
        "Tell me about the French Revolution.",
        "What role did Robespierre play?",
        "How did the Reign of Terror end?",
        "Compare it to the Russian Revolution.",
    ],
]
