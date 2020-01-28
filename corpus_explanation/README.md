## Setup
### Requirements:
#### (Recommended) Virtual environment
```virtualenv env --python=python3.6```
#### Libs
1. pip install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html
2. pip install -r requirements.txt
3. python -m spacy download en
#### Data
1. GloVe 6B - [download](https://nlp.stanford.edu/projects/glove/) and unzip into .vector_cache/
```wget http://nlp.stanford.edu/data/glove.6B.zip```

Optional: 2. IMDB data - [download](http://ai.stanford.edu/~amaas/data/sentiment/) and unzip into .data/
```wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz```