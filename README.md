# Posts

Code, notebooks, and article text for things I write about GenAI, agents, and architecture.
Each folder is one post: the article, the experiment that backs it, and everything needed to
reproduce the numbers in it.

Opinions here are my own and don't represent my employer.

## Index

| post | what it looks at |
|---|---|
| [context-vs-memory](context-vs-memory/) | *Measuring the crossover from context engineering to agent memory.* Where context engineering stops working and agent memory starts being worth its complexity, measured rather than argued |

## Layout

Every post folder follows the same shape:

```
<post-slug>/
├── README.md          what the experiment is, how to run it, known limitations
├── article.md         the article text, as published
├── requirements.txt   dependencies for that post only
├── images/            diagrams and generated charts, embedded in the article
└── src/               the experiment itself, plus its tests
```

Posts are self-contained. Each has its own `requirements.txt`, so nothing here needs a shared
environment, and an old post keeps working when a later one pins something different.

## Running any of them

```bash
cd <post-slug>
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Each post's own README covers what to run from there.
