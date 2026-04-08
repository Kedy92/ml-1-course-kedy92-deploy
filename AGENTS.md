# Practical Machine Learning for Programmers

Notebooks, datasets, homework, and lecture notes for the book "Practical Machine Learning for Programmers" by Utvecklarakademin UA Aktiebolag.

## Hard Rules

- NEVER perform git actions unless explicitly told to
- NEVER modify this AGENTS.md without explicit permission
- NEVER delete or overwrite files in `modified/` folders — they may contain completed examples
- NEVER add "---" to outputs
- Don't overuse " - " when writing
- NEVER estimate time-consumption of approaches

## Repo Structure

```
lessons/                    Lesson notebooks and homework
  <topic>/
    <lesson_number>/
      original/             Starter notebook (clean, never edited by students)
      modified/             Working copy or completed example
    homework/
      original/             Project instructions (empty code cells)
      modified/             Example of a completed project
data/                       Small datasets included in repo
  DATASETS.md               Download instructions for large datasets
lecture_notes/               Per-lesson markdown files (resources, structure, prep notes)
backups/                    Snapshot copies of all notebooks
documentation/              Course plans
```

## Notebook Rules

### Cell Structure

- **Cell 0:** Title + intro (markdown) — frame the lesson, connect to prior knowledge
- **Cell 1:** Hidden helpers (code) — ALL matplotlib/plotting setup here, end with `print("... loaded!")`
- **Cell 2:** Imports (code) — all library imports, suppress warnings, set random seeds
- **Cell 3:** Data path (code) — `DATA_PATH = Path("../../../../data")` with comment to change if needed
- **Remaining cells:** Alternate markdown and code. Never two code cells in a row without explanatory markdown between them

### Data Loading

- Datasets in the repo (`data/` folder): use `DATA_PATH / "dataset_name" / "file.csv"`
- Datasets via torchvision/sklearn: use auto-download with `download=True` and `root='./data'`
- Datasets needing manual download: add a comment with the source and reference `data/DATASETS.md`
- NEVER hardcode absolute paths like `/data/datasets/`, `/opt/ml-datasets/`, or `/home/...`

### Teaching Voice

Write as a colleague explaining concepts over coffee. The notebook IS the teacher — never refer to an author, Claude, or the creation process.

- "Why" before "how" — ask the question before showing the code
- Concrete before abstract — trace real data through the concept, then generalize
- Intuition before code, interpretation after code
- Build incrementally — show one piece, explain it, then add the next
- Normalize not knowing yet — "We'll build this from scratch in Lesson 3"
- Short sentences, contractions, informal but precise
- Max 4-5 sentences per markdown cell, then break
- Avoid bullet points unless listing truly distinct items

### Visualization

- Real matplotlib/plotly graphs only, never text-based data drawings
- ASCII diagrams OK for architecture/flow (neural net structure, pipeline), not for data
- All chart-building code lives in the helpers cell, logic cells call helpers
- Every visualization needs a title and labeled axes

### Notebook Types

**Lesson notebook** — Full guided walkthrough. Every code cell has output. Heavy on prose. The student reads and runs, learning through the narrative.

**Homework notebook** — Scaffolded workspace. Markdown describes what to do, code cells are empty. Phases follow: Understand → Prepare → Train → Evaluate → Iterate. Include dataset options with pros/cons. The `original/` version has empty cells; the `modified/` version may contain a completed example.

## Lecture Notes

Lecture notes live in `lecture_notes/<lesson>/` as markdown files. They are introductory articles that describe what a lesson covers, link to resources, and provide setup help. Think of them as a companion reading — not internal documentation about notebook structure.

- Describe the topics and concepts covered in the lesson
- Link to external resources (videos, articles) organized by timing (before/after lecture)
- Include setup instructions where relevant (Colab, local environment)
- Written for the reader, not about the notebook internals
- Will often contain manually written prose added over time

## Code Style

- Clear, explicit variable names — no acronyms in examples
- Modern Python syntax (f-strings, `|` type unions, Path objects)
- No backwards-compatibility concerns
- Flag outdated techniques when encountered

## Context

- Book plan: `documentation/ml_1_plan.md`
- Target audience: developers with Python/web dev background (FastAPI, Docker, PostgreSQL)
- Primary framework: PyTorch (fastai only in L2 as a high-level preview)
- Dual environment: Google Colab + local with mamba/conda
- The canonical repository is maintained by Utvecklarakademin UA Aktiebolag

## Copyright

All notebooks include a copyright footer:
```html
<div style="text-align: center; color: #888; font-size: 0.85em; margin-top: 40px; padding-top: 10px; border-top: 1px solid #ddd;">
© 2025-2026 Utvecklarakademin UA Aktiebolag. All rights reserved.<br>
This material is proprietary and may not be reproduced, distributed, or shared without written permission.
</div>
```
