# Table of Contents

<!-- toc -->

- [Contributing to LUQ](#contributing-to-luq)
- [Development](#development)
- [Documenting](#documenting)

<!-- tocstop -->

## Contributing to LUQ
Thanks for your interest and willingness to help!

Please, (1) open an issue for a new feature or comment on the existing issue in [the issue tracker](https://github.com/AlexanderVNikitin/luq/issues), (2) open a pull request with the issue (see [the list of pull request](https://github.com/AlexanderVNikitin/luq/pulls)).

The easiest way to make a pull request is to fork the repo, see [GitHub documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).

## Development
To install LUQ in development mode, first install prerequisites:
```bash
pip install -r requirements.txt
```

and then, install LUQ in development mode:
```bash
python setup.py develop
```

To run tests, use pytest, for example:
```bash
pytest tests/test_evals.py::test_accuracy_evaluator_init
```

To run linters, use:
```bash
flake8 luq/
```

## Documenting
We aim to produce high-quality documentation to help our users to use the library. In your contribution, please edit corresponding documentation pages in [./docs](https://github.com/AlexanderVNikitin/luq/tree/main/docs).

