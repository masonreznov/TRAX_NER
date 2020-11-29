## TRAX_NER -- Named Entity Recognition with LSTM network using TRAX library
This repo contains the implementation for named entity recognition (NER) using TRAX library. The implementation is based upon the programming assignment of Natural Language Processing with Sequence Model from [deeplearning.ai](https://www.coursera.org/learn/sequence-models-in-nlp).


### Requirements
- [trax 1.3.x](https://github.com/google/trax)
- [pandas 1.1.x](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)

### File Structure
```
|-TRAX_NER
|   |-data_loader.py -> # contains the data generator function 
|   |-model.py       -> # model definition   
|   |-trainer.py     -> # training loop   
|   |-utils.py       -> # data loading and vocab extraction   
|-README.md
|-train.py           -> # link to the TRAX_NER/trainer.py
```
### Data
- Download this kaggle NER [data](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus). 
- Extract and place it in the `data/` directory.
### Training

```
python train.py \
--output_dir <path/to/output> \
--vocab_size 32000 \
--d_model 512 \
--batch_size 32 \
--train_steps 100
```
- Data loading, vocab extraction and other preliminaries are passed as a pipeline in the `TRAX_NER/trainer.py` file.