# Chicken-Disease-Classification

### 0.0 Note-to-self: Initial steps

1. initiate python -m venv venv >> add activate.bat and template.py
2. Update .gitignore
3. pip install pip-tools >> update requirements.in && requirements-dev.in
4. pip-compile requirements-dev.in >> pip-sync requirements-dev.txt
5. update pyproject.toml to package code, and Dockerfile.
6. pip install -e . >> run pyproject.toml >> package code

## Workflows:

1. update config.yaml
2. update secrets.yaml [Optional]
3. update params.yaml
4. update the entity
5. update the configuration manager in src config
6. update the components
7. update the pipeline
8. update main.py
9. update the dvc.yaml
