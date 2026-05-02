# JetBrains Stem Agent Internship Task

## What this is

A small Python agent that learns how to test a public API and scores its progress.

## Requirements

- Python 3.13
- pip

## Setup

```bash
pip install -r requirements.txt
```

Add your `OPENAI_API_KEY` to the `.env` file.

## How to run

```bash
python main.py
```

## What you will see

The agent explores the API, reflects using OpenAI, evolves a testing strategy, runs tests, and prints a summary with the before score, after score, and improvement.
