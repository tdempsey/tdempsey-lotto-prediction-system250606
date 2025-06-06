# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test/Lint Commands

- Install in dev mode: `pip install -e .`
- Run tests: `python -m unittest discover`
- Run single test: `python -m unittest tests.test_module.TestClass.test_method`
- Generate seed data: `python main.py generate --draws 1000 --numbers 5 --max-number 42 --columns 5`
- Analyze data: `python main.py analyze path/to/data.csv --plot`
- Make predictions: `python main.py predict path/to/data.csv --method statistical`

## Coding Style Guidelines

- Follow PEP 8 conventions
- Use Google-style docstrings with Args and Returns sections
- Use type hints for function parameters and returns
- Organize imports: standard library, then third-party, then local modules
- Handle errors with specific exceptions
- Use 4 spaces for indentation
- Add comprehensive unit tests for new functionality
- Keep functions focused and under 50 lines where possible
- Use descriptive variable names that reflect their purpose