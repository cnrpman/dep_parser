# Dependency Parser

This is a self-implemented dependency parser for 19Fall CSCI-662 @ USC
In consideration of academic integrity, please don't use this code for homework (if any)

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
