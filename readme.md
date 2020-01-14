# Dependency Parser

## Data preparation:
Convert CoNLL to configurations:
* `python preparedata.py [train|dev|test].orig.conll`
Collect dictionary from configurations:
* `python generate_dictionary.py`

## Train model:
* `python train.py`
* please refer to the train.py for options!

## Parse using:
* `python parse.py -m [model file] -i [input file] -o [output file]`