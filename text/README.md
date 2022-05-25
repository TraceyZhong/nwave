This folder contains scripts for experiments on text data. 

Directory structure and selected files

-- code 
    -- experiment.py 
    -- viz_rc.py
    -- distortionComp.py
    -- word_adctivation.ipynb
-- data 
-- result

1. Experiment with GloVe embedding 

To download data, go [this link](https://yaleedu-my.sharepoint.com/:f:/g/personal/xinyi_zhong_yale_edu/EmDi_egy6CxDr7eISstXA4oBzhe-3F3qPpR65lWHyH-AlA?e=fz9ho7) and put all files in folder `data/googleNgram/`

To train the model with GloVe embedding go to the code folder 
```
(code) $ python experiment.py --loader unigram97Loader --sparsity 0.2 --we 30 --wi 5 
``` 
Results will be stored in `result/unigram97Loadertr0.2we30wi5lrW0.1lrA0.1/`.

To observe the organization of tunings in RGB value
```
(code) $ python viz_rc.py --loader unigram97Loader --sparsity 0.2 --we 30 --wi 5 
```

To compare the distortion of t-SNE and our sparse code
```
(code) $ python distortionComp.py 
```

To visualize neuron activation for words, use the jupyter notebook `word_adctivation.ipynb` in `code/`. 

All results can be found in `result/unigram97Loadertr0.2we30wi5lrW0.1lrA0.1/`.

2. Experiment with embedding derived from a LSTM language model. 

To download data, go [this link](https://yaleedu-my.sharepoint.com/:f:/g/personal/xinyi_zhong_yale_edu/Em8bF1nvqddCqyx4o08OUv8Bq_53SbC7o1-6cRgvh4YWqw?e=Ukb7sd) and put all files in folder `data/wiki103/`

To train the model
```
(code) $ python experiment.py --loader fairseqLoader --sparsity 0.2 --we 30 --wi 5 
``` 
Results will be stored in `result/fairseqLoadertr0.2we30wi5lrW0.1lrA0.1/`.

To observe the organization of tunings in RGB value
```
(code) $ viz_rc.py --loader unigram97Loader --sparsity 0.2 --we 30 --wi 5 
```

To visualize neuron activation for words, use the jupyter notebook `word_adctivation.ipynb` in `code/`. 

All results can be found in `result/fairseqLoadertr0.2we30wi5lrW0.1lrA0.1/`.



