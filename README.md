# Atari Model Arena API

Backend API for testing trained models against classic Atari game engines and, when supported, users.

## Requirements

> To load models, view, and interact with Atari gameplay, [atari-model-arena-ui](https://github.com/uni-projects-demos/atari-model-arena-ui/) provides a web interface.

## Install

```bash
pip install -e ".[dev]"
```

## Run

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

## Contribute

Before making a Pull Request, ensure it addresses an Issue, and verify the branch passes:

```bash
code-check
```
