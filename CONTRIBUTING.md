# Contributing to MEMOSCOPE

Thank you for your interest in contributing!

## Setup

```bash
git clone https://github.com/your-org/memoscope
cd memoscope
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

```bash
ruff check .
ruff format .
```

## Adding a New Model Type

1. Add your mock class to `memoscope/core/mock_models.py`
2. Register it in `_MOCK_REGISTRY`
3. Add tests in `tests/test_hooks.py`
4. Update the CLI choices in `memoscope/cli.py`

## Adding a New Metric

1. Implement the metric function in `memoscope/core/hooks.py`
2. Add the field to `StepSnapshot`
3. Compute it inside `MemoryInspector.step()`
4. Add it to `StepSnapshot.to_dict()`
5. Wire it into the frontend in `memoscope/server/templates/index.html`

## Pull Request Checklist

- [ ] Tests pass (`pytest tests/ -v`)
- [ ] Code is linted (`ruff check .`)
- [ ] New features have tests
- [ ] README updated if API changed

## License

By contributing, you agree your code is licensed under MIT.
