# Conda

## Conda Environment

[Miniconda](https://docs.conda.io/en/latest/miniconda.html)

[Conda Cheatsheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)


### Create Environment

```
conda create --name ENV_NAME python=3.9 
```

### Activate Environment
```
conda activate ENV_NAME
```

### Deactivate Environment

```
conda deactivate
```

### List all Environments 
```
conda env list
```


## PiP

```
pip freeze > requirements.txt
```

```
pip install -r requirements.txt
```