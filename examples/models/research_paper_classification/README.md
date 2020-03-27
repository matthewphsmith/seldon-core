# Research Paper Classification for COVID-19 Research

There has been great momentum from the machine learning community to extract insights from the increasingly growing COVID-19 Datasets, such as the Allen Institute for AI [Open Research Dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) as well as the data repository by [Johns Hopkins CSSE](https://github.com/CSSEGISandData/COVID-19).

We believe the most powerful insights can be brought through cross-functional collaboration, such as between infectious disease experts and machine learning practitioners. 

More importantly, once powerful predictive and explanatory models are built, it is especially important to be able to deploy and enable access to these models at scale to power solutions that can solve real-life challenges.

In this small tutorial we will show how you can deploy your machine learning solutions at scale, and we will use a practical example. For this we will be building a simple text classifier using the Allen Institute for AI COVID-19 Open Research Dataset which has been open sourced with over 44,000 scholarly articles on COVID-19, together with the [Arxiv Metadata Research Dataset](https://www.kaggle.com/tayorm/arxiv-papers-metadata) which contains over 1.5M papers.

In this tutorial we will focus primarily around the techniques to productionise an already trained model, and we will showcase how you're able to leverage the Seldon Core Prepackaged Model Servers, the Python Language Wrapper, and some of our AI Explainability infrastructure tools.

## Tutorial Overview

The steps that we will be following in this tutorial include

1) Train and build a simple NLP model with SKLearn and SpaCy

2) Explain your model predictions using Alibi Explain

3) Containerize your model using Seldon Core Language Wrappers and deploy to Kubernetes

5) Test your deployed model by sending requests

6) Deploy our standard Alibi TextExplainer 

7) Test your deployed explainer by sending requests

### Before you start
Make sure you install the following dependencies, as they are critical for this example to work:

* Seldon Core v1.1+ installed with Istio Ingress Enabled ([Documentation Instructions](https://docs.seldon.io/projects/seldon-core/en/latest/workflow/install.html#ingress-support))
* A Kubernetes Cluster with all dependencies specified in the Seldon Core page

Let's get started! 🚀🔥

## 0) Prepare your development environment

First we want to install all the dependencies. 

These are all going to be in our `requirements-dev.txt` file.


```python
%%writefile requirements-dev.txt
scipy>= 0.13.3
scikit-learn>=0.18
spacy==2.0.18
dill==0.2.9
xai==0.0.5
alibi==0.4.0
```

    Overwriting requirements-dev.txt


We can then use pip to install the requirements above:


```python
# Let's first install any dependencies
!pip install -r requirements-dev.txt
```

          Successfully uninstalled matplotlib-3.1.3
      Found existing installation: wrapt 1.12.1
        Uninstalling wrapt-1.12.1:
          Successfully uninstalled wrapt-1.12.1
    Successfully installed alibi-0.4.0 matplotlib-3.0.2 wrapt-1.10.11


Now that everything is installed, we can import all our dependencies.


```python
import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from seldon_core.seldon_client import SeldonClient
import dill
import sys, os
import xai

pd.set_option("display.notebook_repr_html", True)
```

### Let's download the SpaCy English Model


```python
from spacy.cli import download

download("en_core_web_sm")
```

    
    [93m    Linking successful[0m
        /home/alejandro/miniconda3/lib/python3.7/site-packages/en_core_web_sm
        -->
        /home/alejandro/miniconda3/lib/python3.7/site-packages/spacy/data/en_core_web_sm
    
        You can now load the model via spacy.load('en_core_web_sm')
    


## 1) Train and build your NLP model with SKLearn and SpaCy

We can now get started with the training of our model. 

For this tutorial we are going to focus primarily on the productionisation of the model, so we will use a significantly smaller dataset.

More specifically we selected randomly 2000 abstracts from the COVID-19 Open Research Dataset, and 2000 abstracts from the [Arxiv Papers Metadata Dataset](https://www.kaggle.com/tayorm/arxiv-papers-metadata) to compare against to provide a (very simplified) example of an NLP classification model.


```python
df = pd.read_csv("https://raw.githubusercontent.com/axsaucedo/datasets/master/data/research_paper_abstracts.csv")
```

This dataset contains 4000 abstracts randomly picked from the datasets mentioned in the introduction, 2000 COVID-19 related and 2000 nonCOVID19 related


```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>abstract</th>
      <th>is_covid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3995</th>
      <td>This article summarizes current knowledge abou...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3996</th>
      <td>While epidemiological models have traditionall...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3997</th>
      <td>TGEV and PEDV are porcine coronaviruses with t...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3998</th>
      <td>Metagenomics, i.e., the sequencing and analysi...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3999</th>
      <td>Population genetic diversity plays a prominent...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



We can see that we have a distributed set of examples to train and test our model


```python
# Let's see how many examples we have of each class
df["is_covid"].value_counts().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9ba3581910>




![png](README_files/README_14_1.png)


### Split our train test dataset

We first start by splitting our train and test dataset, making sure we have an even breakdown of examples for train test


```python
x = df["abstract"].values
y = df["is_covid"].values
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    stratify=y, 
    random_state=42, 
    test_size=0.1, shuffle=True)
```

### Train our model: Clean Text
As the first step for our model we'll first clean the incoming text data for any less meaningful characters and symbols

For this, we have created a CleanTextTransformer class that will be doing the text pre-processing


```python
from ml_utils import CleanTextTransformer
```

    
    [93m    Linking successful[0m
        /home/alejandro/miniconda3/lib/python3.7/site-packages/en_core_web_sm
        -->
        /home/alejandro/miniconda3/lib/python3.7/site-packages/spacy/data/en_core_web_sm
    
        You can now load the model via spacy.load('en_core_web_sm')
    



```python
# Clean the text
clean_text_transformer = CleanTextTransformer()
x_train_clean = clean_text_transformer.transform(x_train)
```

### Train our model: Tokenize
We now convert our input text into tokens - for this we use the SpaCy module.


```python
from ml_utils import SpacyTokenTransformer
```


```python
# Tokenize the text and get the lemmas
spacy_tokenizer = SpacyTokenTransformer()
x_train_tokenized = spacy_tokenizer.transform(x_train_clean)
```

### Train our model: Vectorize

Now we have to convert our tokens into input our model can read, so we convert our tokens into vector using our TFIDF vectorizer.


```python
# Build tfidf vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=10000,
    preprocessor=lambda x: x, 
    tokenizer=lambda x: x, 
    token_pattern=None,
    ngram_range=(1, 3))

tfidf_vectorizer.fit(x_train_tokenized)
```




    TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.float64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=10000, min_df=1,
            ngram_range=(1, 3), norm='l2',
            preprocessor=<function <lambda> at 0x7f9c0ddce5f0>,
            smooth_idf=True, stop_words=None, strip_accents=None,
            sublinear_tf=False, token_pattern=None,
            tokenizer=<function <lambda> at 0x7f9c0ddce7a0>, use_idf=True,
            vocabulary=None)



Transform our tokens to tfidf vectors


```python
x_train_tfidf = tfidf_vectorizer.transform(
    x_train_tokenized)
```

### Train your model: Prediction

Finally we want to be able to predict using our model. For this, we'll use a logistic regression classifier:


```python
# Train logistic regression classifier
lr = LogisticRegression(C=0.1, solver='sag')
lr.fit(x_train_tfidf, y_train)
```




    LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='warn',
              n_jobs=None, penalty='l2', random_state=None, solver='sag',
              tol=0.0001, verbose=0, warm_start=False)



### Evaluate your model

Now that we've trained our model we can test its performance against our test dataset. 

Let's run a single instance through our classifier first.


```python
def predict_fn(x):
    x_c = clean_text_transformer.transform(x)
    x_s = spacy_tokenizer.transform(x_c)
    x_t = tfidf_vectorizer.transform(x_s)
    return lr.predict(x_t)

print(x_test[0:1])
print(f"Expected class: {y_test[0:1]}")
print(f"Predicted class: {predict_fn(x_test[0:1])}")
```

    ['We report theoretical and simulation studies of phase coexistence in model globular protein solutions, based on short-range, central, pair potential representations of the interaction among macro-particles. After reviewing our previous investigations of hard-core Yukawa and generalised Lennard-Jones potentials, we report more recent results obtained within a DLVO-like description of lysozyme solutions in water and added salt. We show that a one-parameter fit of this model based on Static Light Scattering and Self-Interaction Chromatography data in the dilute protein regime, yields demixing and crystallization curves in good agreement with experimental protein-rich/protein-poor and solubility envelopes. The dependence of cloud and solubility points temperature of the model on the ionic strength is also investigated. Our findings highlight the minimal assumptions on the properties of the microscopic interaction sufficient for a satisfactory reproduction of the phase diagram topology of globular protein solutions.']
    Expected class: [0]
    Predicted class: [0]


Now to evaluate our model we run all our test dataset and extract predictions, so we can evaluate them.


```python
pred = predict_fn(x_test)
```

And now we can see the performance of the predictions, which looks good specifically in this toy dataset.


```python
xai.metrics_plot(y_test, pred)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.994898</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.975000</td>
    </tr>
    <tr>
      <th>specificity</th>
      <td>0.995000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.985000</td>
    </tr>
    <tr>
      <th>auc</th>
      <td>0.985000</td>
    </tr>
    <tr>
      <th>f1</th>
      <td>0.984848</td>
    </tr>
  </tbody>
</table>
</div>




![png](README_files/README_34_1.png)


## 2) Explain your model predictions using Alibi Explain

We will now use the Alibi library to explain predictions from the model we've built

We start by using our alibi explainer for text, and the Spacy NLP module


```python
import spacy
import alibi

nlp = spacy.load("en_core_web_sm", parser=False, entity=False)
```

In order to create a text explainer, we just have to pass the `predict_fn` that we defined above, as it will reverse engineer the explanations


```python
explainer = alibi.explainers.AnchorText(nlp, predict_fn)
```

Now we select a prediction which we will want to explain, in this case we select the index `1` from our test dataset


```python
x_explain = x_test[1]
x_explain
```




    'The modifiable areal unit problem, MAUP, is ever-present although not always appreciated. Through real examples, this article outlines the basic causes of MAUP, namely changes in the size, shape, and/or orientation of spatial categories/polygons used to map areal data. The visual effects of changes to mapped data are obvious even though the impacts on our understanding of the world are profound. The article concludes with a discussion of technical and broader strategic approaches for confronting the effects of MAUP on our treatment and interpretation of areal data.'



And we simply send the prediction request into our explainer, which should take about 10-30 seconds.


```python
explanation = explainer.explain(x_explain, threshold=0.95, use_unk=True)
```

Finally we can print our explanations, which in this case it will tell us the tokens that had the strongest predictive power


```python
print('Anchor: %s' % (' AND '.join(explanation.anchor)))
print('Precision: %.2f' % explanation.precision)
print(f"\nOriginal Sample:\n{x_explain}")
print('\nFirst Example where anchor applies and model predicts is_covid==True')
print(f"\n{explanation.raw['examples'][-1]['covered_true'][0]}".replace("UNK", "___"))
print('\n\nExample where anchor applies and model predicts is_covid==False')
print(f"\n{explanation.raw['examples'][-1]['covered_false'][0]}".replace("UNK", "___"))
```

    Anchor: real
    Precision: 0.97
    
    Original Sample:
    The modifiable areal unit problem, MAUP, is ever-present although not always appreciated. Through real examples, this article outlines the basic causes of MAUP, namely changes in the size, shape, and/or orientation of spatial categories/polygons used to map areal data. The visual effects of changes to mapped data are obvious even though the impacts on our understanding of the world are profound. The article concludes with a discussion of technical and broader strategic approaches for confronting the effects of MAUP on our treatment and interpretation of areal data.
    
    First Example where anchor applies and model predicts is_covid==True
    
    ___ modifiable areal unit ___ , ___ ___ ___ ever - present ___ ___ always ___ . Through real ___ ___ this ___ ___ ___ ___ ___ ___ ___ , namely ___ in the ___ ___ ___ ___ ___ ___ of spatial ___ / ___ used ___ ___ areal data ___ The ___ ___ ___ ___ ___ mapped data are obvious even ___ the impacts ___ our understanding ___ the ___ ___ ___ ___ The ___ concludes with ___ ___ ___ ___ and broader ___ ___ for confronting ___ ___ of MAUP on ___ ___ ___ ___ ___ ___ ___ ___
    
    
    Example where anchor applies and model predicts is_covid==False
    
    ___ modifiable ___ unit ___ ___ MAUP ___ is ever - ___ although ___ ___ appreciated . ___ real ___ , this article outlines ___ basic ___ of MAUP ___ namely changes in the ___ , ___ , and/or orientation of ___ categories ___ polygons used ___ ___ ___ ___ ___ ___ ___ ___ of changes to ___ data are ___ ___ ___ ___ ___ ___ our ___ ___ the ___ ___ profound ___ The article concludes with a discussion of technical ___ broader strategic ___ for confronting the effects of ___ on ___ treatment and ___ ___ ___ ___ ___


## 2) Build your containerized model

Now that we have trained our model, and we know how to use the Alibi explainability library, we can actually deploy these models.

### First use Seldon Core to containerise the model

To get started we will use Seldon to Containerise the model, for this we will export the models we trained above into the current folder


```python
# These are the models we'll deploy
with open('tfidf_vectorizer.model', 'wb') as model_file:
    dill.dump(tfidf_vectorizer, model_file)
with open('lr.model', 'wb') as model_file:
    dill.dump(lr, model_file)
```

The way Seldon Core works from there, is that we need to build a Python Wrapper that exposes the functionality through the `predict` method.

We define a file below called `ResearchClassifier.py` which loads the models we exported and then uses them for any predictions.

Seldon Core will then convert this file into a fully fledged microservice.


```python
%%writefile ResearchClassifier.py

import dill

from ml_utils import CleanTextTransformer, SpacyTokenTransformer

class ResearchClassifier(object):
    def __init__(self):
        
        self._clean_text_transformer = CleanTextTransformer()
        self._spacy_tokenizer = SpacyTokenTransformer()
        
        with open('tfidf_vectorizer.model', 'rb') as model_file:
            self._tfidf_vectorizer = dill.load(model_file)
           
        with open('lr.model', 'rb') as model_file:
            self._lr_model = dill.load(model_file)

    def predict(self, X, feature_names):
        clean_text = self._clean_text_transformer.transform(X)
        spacy_tokens = self._spacy_tokenizer.transform(clean_text)
        tfidf_features = self._tfidf_vectorizer.transform(spacy_tokens)
        predictions = self._lr_model.predict_proba(tfidf_features)
        return predictions


```

    Overwriting ResearchClassifier.py


Before we containerise this Python wrapper, we can test it to make sure it predicts as expected:


```python
from ResearchClassifier import ResearchClassifier

sample = x_test[1:2]

print(f"Input data: {sample}")
print(f"Predicted probabilities for each class: {ResearchClassifier().predict(sample, ['feature_name'])}")
print(f"Actual class: {y_test[1:2]}")
```

    Input data: ['The modifiable areal unit problem, MAUP, is ever-present although not always appreciated. Through real examples, this article outlines the basic causes of MAUP, namely changes in the size, shape, and/or orientation of spatial categories/polygons used to map areal data. The visual effects of changes to mapped data are obvious even though the impacts on our understanding of the world are profound. The article concludes with a discussion of technical and broader strategic approaches for confronting the effects of MAUP on our treatment and interpretation of areal data.']
    Predicted probabilities for each class: [[0.55033599 0.44966401]]
    Actual class: [1]


### Create Docker Image with the S2i utility

Now using the S2I command line interface we wrap our current model to seve it through the Seldon interface

We will need to first define the dependencies that we use in our wrapper into a `requirements.txt` file below:


```python
%%writefile requirements.txt
scipy>= 0.13.3
scikit-learn>=0.18
spacy==2.0.18
dill==0.2.9
```

    Overwriting requirements.txt


We then have to define the environment variables required by Seldon, such as the name of your model file


```python
%%writefile .s2i/environment

MODEL_NAME=ResearchClassifier
API_TYPE=REST
SERVICE_TYPE=MODEL
PERSISTENCE=0
```

    Overwriting .s2i/environment


And now that everything is configured, we can build the `research-classifier:0.1` image using the Seldon Command below - it will take some time depending on internet as it will need to download the docker image, and all the relevant dependencies.


```python
!s2i build . seldonio/seldon-core-s2i-python3:0.18 research-classifier:0.1
```

    tar: scripts: time stamp 2020-03-27 15:37:42.2163223 is 32685.794467881 s in the future
    tar: src/.gitignore: time stamp 2020-03-27 15:37:42.2173814 is 32685.791911573 s in the future
    tar: src/.ipynb_checkpoints/README-checkpoint.ipynb: time stamp 2020-03-27 15:37:42.2195402 is 32685.793796073 s in the future
    tar: src/.ipynb_checkpoints: time stamp 2020-03-27 15:37:42.219017 is 32685.79316194 s in the future
    tar: src/.s2i/environment: time stamp 2020-03-27 15:37:42.2206062 is 32685.794620056 s in the future
    tar: src/.s2i: time stamp 2020-03-27 15:37:42.2206062 is 32685.794555815 s in the future
    tar: src/MyModel.py: time stamp 2020-03-27 15:37:42.2216653 is 32685.795477548 s in the future
    tar: src/README.ipynb: time stamp 2020-03-27 15:37:42.2227778 is 32685.796025306 s in the future
    tar: src/README.md: time stamp 2020-03-27 15:37:42.2238999 is 32685.79699704 s in the future
    tar: src/README_files/README_12_1.png: time stamp 2020-03-27 15:37:42.2244528 is 32685.797339773 s in the future
    tar: src/README_files/README_14_1.png: time stamp 2020-03-27 15:37:42.2255869 is 32685.798348315 s in the future
    tar: src/README_files/README_17_1.png: time stamp 2020-03-27 15:37:42.2261317 is 32685.798792473 s in the future
    tar: src/README_files/README_27_1.png: time stamp 2020-03-27 15:37:42.2272847 is 32685.799836706 s in the future
    tar: src/README_files/README_31_1.png: time stamp 2020-03-27 15:37:42.228415 is 32685.800866581 s in the future
    tar: src/README_files/README_34_1.png: time stamp 2020-03-27 15:37:42.2289955 is 32685.801147323 s in the future
    tar: src/README_files/README_9_1.png: time stamp 2020-03-27 15:37:42.2300969 is 32685.802113848 s in the future
    tar: src/README_files: time stamp 2020-03-27 15:37:42.2300969 is 32685.802056648 s in the future
    tar: src/ResearchClassifier.py: time stamp 2020-03-27 15:37:42.2306202 is 32685.802512998 s in the future
    tar: src/__init__.py: time stamp 2020-03-27 15:37:42.2311656 is 32685.803006615 s in the future
    tar: src/__pycache__/Model.cpython-37.pyc: time stamp 2020-03-27 15:37:42.2353649 is 32685.807060965 s in the future
    tar: src/__pycache__/MyModel.cpython-37.pyc: time stamp 2020-03-27 15:37:42.236397 is 32685.808010081 s in the future
    tar: src/__pycache__/RedditClassifier.cpython-37.pyc: time stamp 2020-03-27 15:37:42.236927 is 32685.808450923 s in the future
    tar: src/__pycache__/ResearchClassifier.cpython-37.pyc: time stamp 2020-03-27 15:37:42.2379643 is 32685.80937144 s in the future
    tar: src/__pycache__/ml_utils.cpython-37.pyc: time stamp 2020-03-27 15:37:42.2385596 is 32685.809895565 s in the future
    tar: src/__pycache__: time stamp 2020-03-27 15:37:42.2385596 is 32685.80984974 s in the future
    tar: src/data/arxiv-abstracts-2k.txt: time stamp 2020-03-27 15:37:42.2412847 is 32685.799745356 s in the future
    tar: src/data/arxiv-papers-metadata/arxiv-abstracts-250k.txt: time stamp 2020-03-27 15:37:42.596636 is 32684.231085172 s in the future
    tar: src/data/arxiv-papers-metadata/arxiv-abstracts-all.txt: time stamp 2020-03-27 15:37:45.5402158 is 32675.816605913 s in the future
    tar: src/data/arxiv-papers-metadata/arxiv-oai-af.tsv: time stamp 2020-03-27 15:37:49.5602117 is 32663.891761419 s in the future
    tar: src/data/arxiv-papers-metadata/arxiv-titles-250k.txt: time stamp 2020-03-27 15:37:49.5750763 is 32663.743165211 s in the future
    tar: src/data/arxiv-papers-metadata/arxiv-titles-all.txt: time stamp 2020-03-27 15:37:49.797229 is 32662.952145468 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/acc-phys.tsv: time stamp 2020-03-27 15:37:49.8475879 is 32663.00146621 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/acc-phys.tsv.xz: time stamp 2020-03-27 15:37:49.8487779 is 32663.00131136 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/adap-org.tsv: time stamp 2020-03-27 15:37:49.8517678 is 32662.997939602 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/adap-org.tsv.xz: time stamp 2020-03-27 15:37:49.8541567 is 32662.998863402 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/alg-geom.tsv: time stamp 2020-03-27 15:37:49.8575667 is 32662.98985721 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/alg-geom.tsv.xz: time stamp 2020-03-27 15:37:49.8591501 is 32662.989495593 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/ao-sci.tsv: time stamp 2020-03-27 15:37:49.8601241 is 32662.990254768 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/ao-sci.tsv.xz: time stamp 2020-03-27 15:37:49.8611272 is 32662.99118626 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/astro-ph.CO.tsv: time stamp 2020-03-27 15:37:50.0066868 is 32662.417080701 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/astro-ph.CO.tsv.xz: time stamp 2020-03-27 15:37:50.0398391 is 32662.310658501 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/astro-ph.EP.tsv: time stamp 2020-03-27 15:37:50.0973786 is 32662.093133059 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/astro-ph.EP.tsv.xz: time stamp 2020-03-27 15:37:50.1113476 is 32662.052629568 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/astro-ph.GA.tsv: time stamp 2020-03-27 15:37:50.2345357 is 32660.7566666 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/astro-ph.GA.tsv.xz: time stamp 2020-03-27 15:37:50.2620966 is 32660.663292292 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/astro-ph.HE.tsv: time stamp 2020-03-27 15:37:50.376371 is 32660.286959691 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/astro-ph.HE.tsv.xz: time stamp 2020-03-27 15:37:50.3996799 is 32660.210640924 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/astro-ph.IM.tsv: time stamp 2020-03-27 15:37:50.4538457 is 32660.051584949 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/astro-ph.IM.tsv.xz: time stamp 2020-03-27 15:37:50.4662755 is 32660.012352749 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/astro-ph.SR.tsv: time stamp 2020-03-27 15:37:50.6552398 is 32659.631082666 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/astro-ph.SR.tsv.xz: time stamp 2020-03-27 15:37:50.8022311 is 32659.649502665 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/astro-ph.tsv: time stamp 2020-03-27 15:37:51.7149834 is 32657.216399465 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/astro-ph.tsv.xz: time stamp 2020-03-27 15:37:51.8770439 is 32656.691736157 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/atom-ph.tsv: time stamp 2020-03-27 15:37:51.921172 is 32656.591844948 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/atom-ph.tsv.xz: time stamp 2020-03-27 15:37:51.9457068 is 32656.585325782 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/bayes-an.tsv: time stamp 2020-03-27 15:37:51.9473443 is 32656.586711407 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/bayes-an.tsv.xz: time stamp 2020-03-27 15:37:51.9489575 is 32656.588229707 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/chao-dyn.tsv: time stamp 2020-03-27 15:37:51.9583045 is 32656.57631109 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/chao-dyn.tsv.xz: time stamp 2020-03-27 15:37:51.9616215 is 32656.574371757 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/chem-ph.tsv: time stamp 2020-03-27 15:37:52.0057209 is 32656.459256365 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/chem-ph.tsv.xz: time stamp 2020-03-27 15:37:52.015118 is 32656.43746744 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cmp-lg.tsv: time stamp 2020-03-27 15:37:52.0181186 is 32656.433116865 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cmp-lg.tsv.xz: time stamp 2020-03-27 15:37:52.020093 is 32656.433119057 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/comp-gas.tsv: time stamp 2020-03-27 15:37:52.0221183 is 32656.433251015 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/comp-gas.tsv.xz: time stamp 2020-03-27 15:37:52.023118 is 32656.433711865 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cond-mat.dis-nn.tsv: time stamp 2020-03-27 15:37:52.0651967 is 32656.286990265 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cond-mat.dis-nn.tsv.xz: time stamp 2020-03-27 15:37:52.0766927 is 32656.258833123 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cond-mat.mes-hall.tsv: time stamp 2020-03-27 15:37:52.227245 is 32655.720420632 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cond-mat.mes-hall.tsv.xz: time stamp 2020-03-27 15:37:52.2578995 is 32655.585732682 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cond-mat.mtrl-sci.tsv: time stamp 2020-03-27 15:37:52.4050173 is 32655.047878965 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cond-mat.mtrl-sci.tsv.xz: time stamp 2020-03-27 15:37:52.4400009 is 32654.94859304 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cond-mat.other.tsv: time stamp 2020-03-27 15:37:52.4709996 is 32654.85067724 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cond-mat.other.tsv.xz: time stamp 2020-03-27 15:37:52.4780325 is 32654.828298074 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cond-mat.quant-gas.tsv: time stamp 2020-03-27 15:37:52.5145749 is 32654.705011649 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cond-mat.quant-gas.tsv.xz: time stamp 2020-03-27 15:37:52.5225898 is 32654.679795949 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cond-mat.soft.tsv: time stamp 2020-03-27 15:37:52.5934852 is 32654.432114324 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cond-mat.soft.tsv.xz: time stamp 2020-03-27 15:37:52.6093984 is 32654.383769066 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cond-mat.stat-mech.tsv: time stamp 2020-03-27 15:37:52.7313074 is 32653.948561332 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cond-mat.stat-mech.tsv.xz: time stamp 2020-03-27 15:37:52.759477 is 32653.860542349 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cond-mat.str-el.tsv: time stamp 2020-03-27 15:37:52.884936 is 32653.427188441 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cond-mat.str-el.tsv.xz: time stamp 2020-03-27 15:37:52.9096393 is 32653.241661591 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cond-mat.supr-con.tsv: time stamp 2020-03-27 15:37:52.9875936 is 32652.960478108 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cond-mat.supr-con.tsv.xz: time stamp 2020-03-27 15:37:53.0049613 is 32652.910525416 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cond-mat.tsv: time stamp 2020-03-27 15:37:54.0048376 is 32650.75166681 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cond-mat.tsv.xz: time stamp 2020-03-27 15:37:54.1255111 is 32650.28731496 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.AI.tsv: time stamp 2020-03-27 15:37:54.1742814 is 32650.114518461 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.AI.tsv.xz: time stamp 2020-03-27 15:37:54.1874015 is 32650.077871744 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.AR.tsv: time stamp 2020-03-27 15:37:54.192786 is 32650.067747461 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.AR.tsv.xz: time stamp 2020-03-27 15:37:54.1949738 is 32650.066223544 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.CC.tsv: time stamp 2020-03-27 15:37:54.2129414 is 32649.986687569 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.CC.tsv.xz: time stamp 2020-03-27 15:37:54.218155 is 32649.970518986 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.CE.tsv: time stamp 2020-03-27 15:37:54.2277106 is 32649.939554669 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.CE.tsv.xz: time stamp 2020-03-27 15:37:54.2313448 is 32649.931705211 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.CG.tsv: time stamp 2020-03-27 15:37:54.2419026 is 32649.889916561 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.CG.tsv.xz: time stamp 2020-03-27 15:37:54.2458037 is 32649.881603619 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.CL.tsv: time stamp 2020-03-27 15:37:54.281109 is 32649.721328469 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.CL.tsv.xz: time stamp 2020-03-27 15:37:54.290713 is 32649.694503869 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.CR.tsv: time stamp 2020-03-27 15:37:54.3236457 is 32649.578612136 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.CR.tsv.xz: time stamp 2020-03-27 15:37:54.3340004 is 32649.546331061 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.CV.tsv: time stamp 2020-03-27 15:37:54.4297162 is 32649.268090378 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.CV.tsv.xz: time stamp 2020-03-27 15:37:54.4482008 is 32649.20064942 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.CY.tsv: time stamp 2020-03-27 15:37:54.4681284 is 32649.13673157 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.CY.tsv.xz: time stamp 2020-03-27 15:37:54.4742129 is 32649.119869337 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.DB.tsv: time stamp 2020-03-27 15:37:54.4857561 is 32649.081941037 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.DB.tsv.xz: time stamp 2020-03-27 15:37:54.4899261 is 32649.074898862 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.DC.tsv: time stamp 2020-03-27 15:37:54.5158523 is 32648.979069229 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.DC.tsv.xz: time stamp 2020-03-27 15:37:54.5237358 is 32648.959984662 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.DL.tsv: time stamp 2020-03-27 15:37:54.531616 is 32648.936057112 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.DL.tsv.xz: time stamp 2020-03-27 15:37:54.5343722 is 32648.93131372 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.DM.tsv: time stamp 2020-03-27 15:37:54.5524455 is 32648.871703245 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.DM.tsv.xz: time stamp 2020-03-27 15:37:54.5583013 is 32648.859340954 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.DS.tsv: time stamp 2020-03-27 15:37:54.5969697 is 32648.726535562 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.DS.tsv.xz: time stamp 2020-03-27 15:37:54.6080591 is 32648.696220504 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.ET.tsv: time stamp 2020-03-27 15:37:54.6140647 is 32648.678137646 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.ET.tsv.xz: time stamp 2020-03-27 15:37:54.6172591 is 32648.675701529 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.FL.tsv: time stamp 2020-03-27 15:37:54.6242811 is 32648.651206004 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.FL.tsv.xz: time stamp 2020-03-27 15:37:54.6269173 is 32648.647038221 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.GL.tsv: time stamp 2020-03-27 15:37:54.6285097 is 32648.647010171 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.GL.tsv.xz: time stamp 2020-03-27 15:37:54.6301115 is 32648.647733062 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.GR.tsv: time stamp 2020-03-27 15:37:54.6355487 is 32648.629894954 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.GR.tsv.xz: time stamp 2020-03-27 15:37:54.637719 is 32648.627384387 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.GT.tsv: time stamp 2020-03-27 15:37:54.6581353 is 32648.579488287 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.GT.tsv.xz: time stamp 2020-03-27 15:37:54.6649901 is 32648.572692329 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.HC.tsv: time stamp 2020-03-27 15:37:54.6830307 is 32648.533297304 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.HC.tsv.xz: time stamp 2020-03-27 15:37:54.6872998 is 32648.526059237 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.IR.tsv: time stamp 2020-03-27 15:37:54.7048987 is 32648.471013921 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.IR.tsv.xz: time stamp 2020-03-27 15:37:54.7110814 is 32648.458951229 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.IT.tsv: time stamp 2020-03-27 15:37:54.7814604 is 32648.196587538 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.IT.tsv.xz: time stamp 2020-03-27 15:37:54.7973705 is 32648.138303279 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.LG.tsv: time stamp 2020-03-27 15:37:54.8971936 is 32647.787132546 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.LG.tsv.xz: time stamp 2020-03-27 15:37:54.9202942 is 32647.705963072 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.LO.tsv: time stamp 2020-03-27 15:37:54.9421584 is 32647.632262647 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.LO.tsv.xz: time stamp 2020-03-27 15:37:54.9481914 is 32647.61765988 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.MA.tsv: time stamp 2020-03-27 15:37:54.9555525 is 32647.590537722 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.MA.tsv.xz: time stamp 2020-03-27 15:37:54.9597606 is 32647.587156747 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.MM.tsv: time stamp 2020-03-27 15:37:54.9671545 is 32647.568244822 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.MM.tsv.xz: time stamp 2020-03-27 15:37:54.9702865 is 32647.563669988 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.MS.tsv: time stamp 2020-03-27 15:37:54.9755014 is 32647.555605822 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.MS.tsv.xz: time stamp 2020-03-27 15:37:54.9781046 is 32647.554316063 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.NA.tsv: time stamp 2020-03-27 15:37:54.9897587 is 32647.524697872 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.NA.tsv.xz: time stamp 2020-03-27 15:37:54.9934072 is 32647.51825588 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.NE.tsv: time stamp 2020-03-27 15:37:55.0123774 is 32647.452267614 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.NE.tsv.xz: time stamp 2020-03-27 15:37:55.0181572 is 32647.441590064 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.NI.tsv: time stamp 2020-03-27 15:37:55.0547722 is 32647.330581289 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.NI.tsv.xz: time stamp 2020-03-27 15:37:55.0637694 is 32647.30679823 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.OH.tsv: time stamp 2020-03-27 15:37:55.0685668 is 32647.292656472 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.OH.tsv.xz: time stamp 2020-03-27 15:37:55.0712017 is 32647.290006439 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.OS.tsv: time stamp 2020-03-27 15:37:55.0733327 is 32647.28512383 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.OS.tsv.xz: time stamp 2020-03-27 15:37:55.0749308 is 32647.284822089 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.PF.tsv: time stamp 2020-03-27 15:37:55.0812026 is 32647.26092533 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.PF.tsv.xz: time stamp 2020-03-27 15:37:55.0833332 is 32647.255932905 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.PL.tsv: time stamp 2020-03-27 15:37:55.0950874 is 32647.204012239 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.PL.tsv.xz: time stamp 2020-03-27 15:37:55.0983084 is 32647.190653039 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.RO.tsv: time stamp 2020-03-27 15:37:55.1172917 is 32647.098633997 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.RO.tsv.xz: time stamp 2020-03-27 15:37:55.1233139 is 32647.084062464 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.SC.tsv: time stamp 2020-03-27 15:37:55.127751 is 32647.069866447 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.SC.tsv.xz: time stamp 2020-03-27 15:37:55.1298193 is 32647.068957639 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.SD.tsv: time stamp 2020-03-27 15:37:55.1377104 is 32647.041519831 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.SD.tsv.xz: time stamp 2020-03-27 15:37:55.1404237 is 32647.033275864 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.SE.tsv: time stamp 2020-03-27 15:37:55.1564355 is 32646.967456239 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.SE.tsv.xz: time stamp 2020-03-27 15:37:55.1622258 is 32646.951792964 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.SI.tsv: time stamp 2020-03-27 15:37:55.1914443 is 32646.832484081 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.SI.tsv.xz: time stamp 2020-03-27 15:37:55.1988387 is 32646.804569939 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.SY.tsv: time stamp 2020-03-27 15:37:55.2213677 is 32646.701453181 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/cs.SY.tsv.xz: time stamp 2020-03-27 15:37:55.2282864 is 32646.685325373 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/dg-ga.tsv: time stamp 2020-03-27 15:37:55.2313792 is 32646.683328564 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/dg-ga.tsv.xz: time stamp 2020-03-27 15:37:55.2324124 is 32646.683458048 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/econ.EM.tsv: time stamp 2020-03-27 15:37:55.2360612 is 32646.676445548 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/econ.EM.tsv.xz: time stamp 2020-03-27 15:37:55.2381557 is 32646.675959548 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/econ.GN.tsv: time stamp 2020-03-27 15:37:55.2402376 is 32646.672766914 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/econ.GN.tsv.xz: time stamp 2020-03-27 15:37:55.2417999 is 32646.672782864 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/econ.TH.tsv: time stamp 2020-03-27 15:37:55.244016 is 32646.672545106 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/econ.TH.tsv.xz: time stamp 2020-03-27 15:37:55.2451363 is 32646.672879556 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/eess.AS.tsv: time stamp 2020-03-27 15:37:55.2518731 is 32646.656081264 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/eess.AS.tsv.xz: time stamp 2020-03-27 15:37:55.2546328 is 32646.653458314 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/eess.IV.tsv: time stamp 2020-03-27 15:37:55.2637969 is 32646.634323906 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/eess.IV.tsv.xz: time stamp 2020-03-27 15:37:55.2669326 is 32646.630647173 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/eess.SP.tsv: time stamp 2020-03-27 15:37:55.2814887 is 32646.584339331 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/eess.SP.tsv.xz: time stamp 2020-03-27 15:37:55.2862218 is 32646.574900914 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/eess.SY.tsv: time stamp 2020-03-27 15:37:55.2898941 is 32646.569870256 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/eess.SY.tsv.xz: time stamp 2020-03-27 15:37:55.2920476 is 32646.568625881 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/funct-an.tsv: time stamp 2020-03-27 15:37:55.2941654 is 32646.568089623 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/funct-an.tsv.xz: time stamp 2020-03-27 15:37:55.2952046 is 32646.56844329 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/gr-qc.tsv: time stamp 2020-03-27 15:37:55.4557208 is 32645.967560865 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/gr-qc.tsv.xz: time stamp 2020-03-27 15:37:55.4923916 is 32645.854835341 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/hep-ex.tsv: time stamp 2020-03-27 15:37:55.5939033 is 32645.488149116 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/hep-ex.tsv.xz: time stamp 2020-03-27 15:37:55.6124287 is 32645.423080833 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/hep-lat.tsv: time stamp 2020-03-27 15:37:55.658046 is 32645.239220833 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/hep-lat.tsv.xz: time stamp 2020-03-27 15:37:55.669058 is 32645.206699083 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/hep-ph.tsv: time stamp 2020-03-27 15:37:55.9484086 is 32644.007546793 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/hep-ph.tsv.xz: time stamp 2020-03-27 15:37:56.0049724 is 32643.653673402 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/hep-th.tsv: time stamp 2020-03-27 15:37:56.5614699 is 32642.858661654 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/hep-th.tsv.xz: time stamp 2020-03-27 15:37:56.6106951 is 32642.670259421 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math-ph.tsv: time stamp 2020-03-27 15:37:56.7314613 is 32642.258876579 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math-ph.tsv.xz: time stamp 2020-03-27 15:37:56.7594713 is 32642.179829196 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.AC.tsv: time stamp 2020-03-27 15:37:56.7777635 is 32642.131774488 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.AC.tsv.xz: time stamp 2020-03-27 15:37:56.78281 is 32642.123437055 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.AG.tsv: time stamp 2020-03-27 15:37:56.8518892 is 32641.93188348 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.AG.tsv.xz: time stamp 2020-03-27 15:37:56.8681377 is 32641.896012705 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.AP.tsv: time stamp 2020-03-27 15:37:56.9386559 is 32641.680618439 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.AP.tsv.xz: time stamp 2020-03-27 15:37:56.9559823 is 32641.638436039 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.AT.tsv: time stamp 2020-03-27 15:37:56.9760385 is 32641.577965147 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.AT.tsv.xz: time stamp 2020-03-27 15:37:56.9810378 is 32641.565070706 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.CA.tsv: time stamp 2020-03-27 15:37:57.0090383 is 32641.459136106 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.CA.tsv.xz: time stamp 2020-03-27 15:37:57.0170375 is 32641.441231289 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.CO.tsv: time stamp 2020-03-27 15:37:57.0960383 is 32641.168085056 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.CO.tsv.xz: time stamp 2020-03-27 15:37:57.1150379 is 32641.091091465 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.CT.tsv: time stamp 2020-03-27 15:37:57.126038 is 32641.050121248 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.CT.tsv.xz: time stamp 2020-03-27 15:37:57.1300714 is 32641.04289199 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.CV.tsv: time stamp 2020-03-27 15:37:57.150073 is 32640.97000599 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.CV.tsv.xz: time stamp 2020-03-27 15:37:57.155073 is 32640.957778281 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.DG.tsv: time stamp 2020-03-27 15:37:57.2035057 is 32640.787318823 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.DG.tsv.xz: time stamp 2020-03-27 15:37:57.2138582 is 32640.75230544 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.DS.tsv: time stamp 2020-03-27 15:37:57.2592956 is 32640.59506114 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.DS.tsv.xz: time stamp 2020-03-27 15:37:57.2705076 is 32640.56309429 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.FA.tsv: time stamp 2020-03-27 15:37:57.3060977 is 32640.451108174 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.FA.tsv.xz: time stamp 2020-03-27 15:37:57.3143847 is 32640.429880624 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.GM.tsv: time stamp 2020-03-27 15:37:57.3196091 is 32640.418526449 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.GM.tsv.xz: time stamp 2020-03-27 15:37:57.3228647 is 32640.416114807 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.GN.tsv: time stamp 2020-03-27 15:37:57.3303575 is 32640.399178116 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.GN.tsv.xz: time stamp 2020-03-27 15:37:57.3335109 is 32640.396582157 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.GR.tsv: time stamp 2020-03-27 15:37:57.3598341 is 32640.315698583 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.GR.tsv.xz: time stamp 2020-03-27 15:37:57.3662037 is 32640.296805066 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.GT.tsv: time stamp 2020-03-27 15:37:57.4056927 is 32640.219738124 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.GT.tsv.xz: time stamp 2020-03-27 15:37:57.4156989 is 32640.202130483 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.HO.tsv: time stamp 2020-03-27 15:37:57.4214135 is 32640.190143416 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.HO.tsv.xz: time stamp 2020-03-27 15:37:57.4234908 is 32640.186841425 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.IT.tsv: time stamp 2020-03-27 15:37:57.4928171 is 32639.920026617 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.IT.tsv.xz: time stamp 2020-03-27 15:37:57.5087434 is 32639.868565034 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.KT.tsv: time stamp 2020-03-27 15:37:57.5166167 is 32639.844081225 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.KT.tsv.xz: time stamp 2020-03-27 15:37:57.5192673 is 32639.840122084 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.LO.tsv: time stamp 2020-03-27 15:37:57.5331773 is 32639.779303317 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.LO.tsv.xz: time stamp 2020-03-27 15:37:57.5374843 is 32639.768855209 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.MG.tsv: time stamp 2020-03-27 15:37:57.5504175 is 32639.721073317 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.MG.tsv.xz: time stamp 2020-03-27 15:37:57.5547155 is 32639.711448642 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.MP.tsv: time stamp 2020-03-27 15:37:57.6633233 is 32639.29349491 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.MP.tsv.xz: time stamp 2020-03-27 15:37:57.6880275 is 32639.207778519 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.NA.tsv: time stamp 2020-03-27 15:37:57.7285807 is 32639.059888811 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.NA.tsv.xz: time stamp 2020-03-27 15:37:57.7379707 is 32639.029252569 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.NT.tsv: time stamp 2020-03-27 15:37:57.7770139 is 32638.870985344 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.NT.tsv.xz: time stamp 2020-03-27 15:37:57.7866479 is 32638.842320694 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.OA.tsv: time stamp 2020-03-27 15:37:57.8039389 is 32638.786847986 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.OA.tsv.xz: time stamp 2020-03-27 15:37:57.8086644 is 32638.774685836 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.OC.tsv: time stamp 2020-03-27 15:37:57.858638 is 32638.595585245 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.OC.tsv.xz: time stamp 2020-03-27 15:37:57.8706393 is 32638.54832572 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.PR.tsv: time stamp 2020-03-27 15:37:57.9374646 is 32638.210646163 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.PR.tsv.xz: time stamp 2020-03-27 15:37:57.9529152 is 32638.149347363 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.QA.tsv: time stamp 2020-03-27 15:37:57.9800192 is 32638.033771055 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.QA.tsv.xz: time stamp 2020-03-27 15:37:57.986824 is 32638.017003171 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.RA.tsv: time stamp 2020-03-27 15:37:58.0061614 is 32637.952000038 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.RA.tsv.xz: time stamp 2020-03-27 15:37:58.0125169 is 32637.940450855 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.RT.tsv: time stamp 2020-03-27 15:37:58.0512271 is 32637.844617772 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.RT.tsv.xz: time stamp 2020-03-27 15:37:58.0603842 is 32637.82793748 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.SG.tsv: time stamp 2020-03-27 15:37:58.0729001 is 32637.790198955 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.SG.tsv.xz: time stamp 2020-03-27 15:37:58.0765602 is 32637.781707739 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.SP.tsv: time stamp 2020-03-27 15:37:58.0892054 is 32637.738849547 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.SP.tsv.xz: time stamp 2020-03-27 15:37:58.0932126 is 32637.73091213 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.ST.tsv: time stamp 2020-03-27 15:37:58.127596 is 32637.625698822 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/math.ST.tsv.xz: time stamp 2020-03-27 15:37:58.1363707 is 32637.605053656 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/mtrl-th.tsv: time stamp 2020-03-27 15:37:58.1386302 is 32637.604948456 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/mtrl-th.tsv.xz: time stamp 2020-03-27 15:37:58.1402326 is 32637.605519414 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/nlin.AO.tsv: time stamp 2020-03-27 15:37:58.152523 is 32637.563008206 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/nlin.AO.tsv.xz: time stamp 2020-03-27 15:37:58.1567809 is 32637.553006081 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/nlin.CD.tsv: time stamp 2020-03-27 15:37:58.1839591 is 32637.462500589 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/nlin.CD.tsv.xz: time stamp 2020-03-27 15:37:58.1916676 is 32637.440096323 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/nlin.CG.tsv: time stamp 2020-03-27 15:37:58.1953452 is 32637.433155414 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/nlin.CG.tsv.xz: time stamp 2020-03-27 15:37:58.1973452 is 32637.432520964 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/nlin.PS.tsv: time stamp 2020-03-27 15:37:58.2158338 is 32637.383232981 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/nlin.PS.tsv.xz: time stamp 2020-03-27 15:37:58.2217881 is 32637.373021456 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/nlin.SI.tsv: time stamp 2020-03-27 15:37:58.2392916 is 32637.309872956 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/nlin.SI.tsv.xz: time stamp 2020-03-27 15:37:58.2441169 is 32637.29839019 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/nucl-ex.tsv: time stamp 2020-03-27 15:37:58.2890722 is 32637.104517724 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/nucl-ex.tsv.xz: time stamp 2020-03-27 15:37:58.2993197 is 32637.069442457 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/nucl-th.tsv: time stamp 2020-03-27 15:37:58.3980193 is 32636.725819008 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/nucl-th.tsv.xz: time stamp 2020-03-27 15:37:58.4188176 is 32636.662493658 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/patt-sol.tsv: time stamp 2020-03-27 15:37:58.4215383 is 32636.658774808 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/patt-sol.tsv.xz: time stamp 2020-03-27 15:37:58.4232653 is 32636.659246008 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.acc-ph.tsv: time stamp 2020-03-27 15:37:58.4354323 is 32636.618906008 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.acc-ph.tsv.xz: time stamp 2020-03-27 15:37:58.4398005 is 32636.609790066 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.ao-ph.tsv: time stamp 2020-03-27 15:37:58.4501424 is 32636.574721125 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.ao-ph.tsv.xz: time stamp 2020-03-27 15:37:58.4538146 is 32636.567491092 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.app-ph.tsv: time stamp 2020-03-27 15:37:58.4676593 is 32636.518336925 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.app-ph.tsv.xz: time stamp 2020-03-27 15:37:58.4724764 is 32636.50587735 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.atm-clus.tsv: time stamp 2020-03-27 15:37:58.4788475 is 32636.488185667 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.atm-clus.tsv.xz: time stamp 2020-03-27 15:37:58.4816502 is 32636.4857199 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.atom-ph.tsv: time stamp 2020-03-27 15:37:58.5183058 is 32636.347670775 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.atom-ph.tsv.xz: time stamp 2020-03-27 15:37:58.5277481 is 32636.285753476 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.bio-ph.tsv: time stamp 2020-03-27 15:37:58.5571337 is 32636.080226476 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.bio-ph.tsv.xz: time stamp 2020-03-27 15:37:58.5654049 is 32636.027961659 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.chem-ph.tsv: time stamp 2020-03-27 15:37:58.601044 is 32635.889333485 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.chem-ph.tsv.xz: time stamp 2020-03-27 15:37:58.6108568 is 32635.865348885 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.class-ph.tsv: time stamp 2020-03-27 15:37:58.6256569 is 32635.817059443 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.class-ph.tsv.xz: time stamp 2020-03-27 15:37:58.6304971 is 32635.807179852 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.comp-ph.tsv: time stamp 2020-03-27 15:37:58.6612491 is 32635.696468235 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.comp-ph.tsv.xz: time stamp 2020-03-27 15:37:58.6698495 is 32635.671134085 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.data-an.tsv: time stamp 2020-03-27 15:37:58.6896071 is 32635.60679111 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.data-an.tsv.xz: time stamp 2020-03-27 15:37:58.6964277 is 32635.59529081 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.ed-ph.tsv: time stamp 2020-03-27 15:37:58.7048608 is 32635.573359269 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.ed-ph.tsv.xz: time stamp 2020-03-27 15:37:58.7079522 is 32635.569509102 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.flu-dyn.tsv: time stamp 2020-03-27 15:37:58.7737352 is 32635.448767236 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.flu-dyn.tsv.xz: time stamp 2020-03-27 15:37:58.801755 is 32635.438039128 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.gen-ph.tsv: time stamp 2020-03-27 15:37:58.851985 is 32635.407621244 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.gen-ph.tsv.xz: time stamp 2020-03-27 15:37:58.8701608 is 32635.406677944 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.geo-ph.tsv: time stamp 2020-03-27 15:37:58.9063362 is 32635.391459578 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.geo-ph.tsv.xz: time stamp 2020-03-27 15:37:58.9204847 is 32635.389678095 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.hist-ph.tsv: time stamp 2020-03-27 15:37:58.9378316 is 32635.37351457 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.hist-ph.tsv.xz: time stamp 2020-03-27 15:37:58.9488313 is 32635.375371261 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.ins-det.tsv: time stamp 2020-03-27 15:37:59.0437617 is 32635.30757427 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.ins-det.tsv.xz: time stamp 2020-03-27 15:37:59.0659506 is 32635.291619978 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.med-ph.tsv: time stamp 2020-03-27 15:37:59.0895164 is 32635.270927495 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.med-ph.tsv.xz: time stamp 2020-03-27 15:37:59.0979669 is 32635.268291162 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.optics.tsv: time stamp 2020-03-27 15:37:59.2184476 is 32635.088327346 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.optics.tsv.xz: time stamp 2020-03-27 15:37:59.2344409 is 32635.044541554 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.plasm-ph.tsv: time stamp 2020-03-27 15:37:59.2676109 is 32634.961729021 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.plasm-ph.tsv.xz: time stamp 2020-03-27 15:37:59.2780162 is 32634.945611829 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.pop-ph.tsv: time stamp 2020-03-27 15:37:59.2863315 is 32634.936243988 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.pop-ph.tsv.xz: time stamp 2020-03-27 15:37:59.2896258 is 32634.935033896 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.soc-ph.tsv: time stamp 2020-03-27 15:37:59.329269 is 32634.780885013 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.soc-ph.tsv.xz: time stamp 2020-03-27 15:37:59.3392347 is 32634.747395938 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.space-ph.tsv: time stamp 2020-03-27 15:37:59.3522334 is 32634.708768822 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/physics.space-ph.tsv.xz: time stamp 2020-03-27 15:37:59.3582482 is 32634.701767155 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/plasm-ph.tsv: time stamp 2020-03-27 15:37:59.388347 is 32634.611561355 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/plasm-ph.tsv.xz: time stamp 2020-03-27 15:37:59.3946245 is 32634.591142814 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-alg.tsv: time stamp 2020-03-27 15:37:59.3982719 is 32634.584370064 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-alg.tsv.xz: time stamp 2020-03-27 15:37:59.399819 is 32634.582114014 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-bio.BM.tsv: time stamp 2020-03-27 15:37:59.4106558 is 32634.547530547 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-bio.BM.tsv.xz: time stamp 2020-03-27 15:37:59.41443 is 32634.539863664 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-bio.CB.tsv: time stamp 2020-03-27 15:37:59.4207192 is 32634.527137764 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-bio.CB.tsv.xz: time stamp 2020-03-27 15:37:59.42278 is 32634.524412289 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-bio.GN.tsv: time stamp 2020-03-27 15:37:59.4297241 is 32634.505282506 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-bio.GN.tsv.xz: time stamp 2020-03-27 15:37:59.4325161 is 32634.501230656 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-bio.MN.tsv: time stamp 2020-03-27 15:37:59.441773 is 32634.474119297 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-bio.MN.tsv.xz: time stamp 2020-03-27 15:37:59.4460069 is 32634.470092997 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-bio.NC.tsv: time stamp 2020-03-27 15:37:59.4617597 is 32634.417141014 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-bio.NC.tsv.xz: time stamp 2020-03-27 15:37:59.466623 is 32634.404329189 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-bio.OT.tsv: time stamp 2020-03-27 15:37:59.4702857 is 32634.398460723 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-bio.OT.tsv.xz: time stamp 2020-03-27 15:37:59.471896 is 32634.396400906 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-bio.PE.tsv: time stamp 2020-03-27 15:37:59.4910438 is 32634.329366181 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-bio.PE.tsv.xz: time stamp 2020-03-27 15:37:59.4969665 is 32634.312702789 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-bio.QM.tsv: time stamp 2020-03-27 15:37:59.5129098 is 32634.253680398 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-bio.QM.tsv.xz: time stamp 2020-03-27 15:37:59.518518 is 32634.234756406 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-bio.SC.tsv: time stamp 2020-03-27 15:37:59.5233054 is 32634.216519831 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-bio.SC.tsv.xz: time stamp 2020-03-27 15:37:59.5253798 is 32634.21371999 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-bio.TO.tsv: time stamp 2020-03-27 15:37:59.5302138 is 32634.199287448 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-bio.TO.tsv.xz: time stamp 2020-03-27 15:37:59.532827 is 32634.197299948 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-bio.tsv: time stamp 2020-03-27 15:37:59.6063701 is 32633.920094057 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-bio.tsv.xz: time stamp 2020-03-27 15:37:59.6253485 is 32633.854216691 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-fin.CP.tsv: time stamp 2020-03-27 15:37:59.6304305 is 32633.847251158 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-fin.CP.tsv.xz: time stamp 2020-03-27 15:37:59.6326197 is 32633.846058191 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-fin.EC.tsv: time stamp 2020-03-27 15:37:59.6364063 is 32633.839090433 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-fin.EC.tsv.xz: time stamp 2020-03-27 15:37:59.6385695 is 32633.838487883 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-fin.GN.tsv: time stamp 2020-03-27 15:37:59.6445149 is 32633.821736799 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-fin.GN.tsv.xz: time stamp 2020-03-27 15:37:59.6467044 is 32633.818884033 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-fin.MF.tsv: time stamp 2020-03-27 15:37:59.6503652 is 32633.812108541 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-fin.MF.tsv.xz: time stamp 2020-03-27 15:37:59.6519618 is 32633.811677574 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-fin.PM.tsv: time stamp 2020-03-27 15:37:59.6554125 is 32633.802318616 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-fin.PM.tsv.xz: time stamp 2020-03-27 15:37:59.6569661 is 32633.801373074 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-fin.PR.tsv: time stamp 2020-03-27 15:37:59.6612165 is 32633.791916599 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-fin.PR.tsv.xz: time stamp 2020-03-27 15:37:59.6644611 is 32633.792183224 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-fin.RM.tsv: time stamp 2020-03-27 15:37:59.6697035 is 32633.782513649 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-fin.RM.tsv.xz: time stamp 2020-03-27 15:37:59.6729364 is 32633.782302524 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-fin.ST.tsv: time stamp 2020-03-27 15:37:59.6804853 is 32633.766244833 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-fin.ST.tsv.xz: time stamp 2020-03-27 15:37:59.6836262 is 32633.76273385 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-fin.TR.tsv: time stamp 2020-03-27 15:37:59.6879464 is 32633.755948583 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/q-fin.TR.tsv.xz: time stamp 2020-03-27 15:37:59.6901238 is 32633.755014541 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/quant-ph.tsv: time stamp 2020-03-27 15:37:59.8628438 is 32632.937447627 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/quant-ph.tsv.xz: time stamp 2020-03-27 15:37:59.9078411 is 32632.784999844 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/solv-int.tsv: time stamp 2020-03-27 15:37:59.9117014 is 32632.778104052 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/solv-int.tsv.xz: time stamp 2020-03-27 15:37:59.9139876 is 32632.778020677 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/stat.AP.tsv: time stamp 2020-03-27 15:37:59.9368793 is 32632.691195844 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/stat.AP.tsv.xz: time stamp 2020-03-27 15:37:59.9449725 is 32632.667367436 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/stat.CO.tsv: time stamp 2020-03-27 15:37:59.956093 is 32632.62259942 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/stat.CO.tsv.xz: time stamp 2020-03-27 15:37:59.959893 is 32632.617406561 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/stat.ME.tsv: time stamp 2020-03-27 15:37:59.9901502 is 32632.511588103 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/stat.ME.tsv.xz: time stamp 2020-03-27 15:37:59.9975145 is 32632.489456703 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/stat.ML.tsv: time stamp 2020-03-27 15:38:00.0668199 is 32632.192570846 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/stat.ML.tsv.xz: time stamp 2020-03-27 15:38:00.0848141 is 32632.127541088 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/stat.OT.tsv: time stamp 2020-03-27 15:38:00.0878152 is 32632.120362104 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/stat.OT.tsv.xz: time stamp 2020-03-27 15:38:00.0898148 is 32632.119712954 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/stat.TH.tsv: time stamp 2020-03-27 15:38:00.1392752 is 32632.027583963 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/stat.TH.tsv.xz: time stamp 2020-03-27 15:38:00.1486556 is 32632.008954238 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/supr-con.tsv: time stamp 2020-03-27 15:38:00.2424462 is 32631.620515798 s in the future
    tar: src/data/arxiv-papers-metadata/per_category/supr-con.tsv.xz: time stamp 2020-03-27 15:38:00.2624474 is 32631.543829173 s in the future
    tar: src/data/arxiv-papers-metadata/per_category: time stamp 2020-03-27 15:38:00.2435583 is 32631.524777898 s in the future
    tar: src/data/arxiv-papers-metadata/per_year/1993.tsv: time stamp 2020-03-27 15:38:00.2830493 is 32631.492379331 s in the future
    tar: src/data/arxiv-papers-metadata/per_year/1994.tsv: time stamp 2020-03-27 15:38:00.3029648 is 32631.389720973 s in the future
    tar: src/data/arxiv-papers-metadata/per_year/1995.tsv: time stamp 2020-03-27 15:38:00.3283188 is 32631.301509249 s in the future
    tar: src/data/arxiv-papers-metadata/per_year/1996.tsv: time stamp 2020-03-27 15:38:00.3601562 is 32631.19829279 s in the future
    tar: src/data/arxiv-papers-metadata/per_year/1997.tsv: time stamp 2020-03-27 15:38:00.3990186 is 32631.063540299 s in the future
    tar: src/data/arxiv-papers-metadata/per_year/1998.tsv: time stamp 2020-03-27 15:38:00.4447058 is 32630.890771925 s in the future
    tar: src/data/arxiv-papers-metadata/per_year/1999.tsv: time stamp 2020-03-27 15:38:00.4994974 is 32630.697579534 s in the future
    tar: src/data/arxiv-papers-metadata/per_year/2000.tsv: time stamp 2020-03-27 15:38:00.5627499 is 32630.483801926 s in the future
    tar: src/data/arxiv-papers-metadata/per_year/2001.tsv: time stamp 2020-03-27 15:38:00.62597 is 32630.253961835 s in the future
    tar: src/data/arxiv-papers-metadata/per_year/2002.tsv: time stamp 2020-03-27 15:38:00.6988459 is 32629.998183544 s in the future
    tar: src/data/arxiv-papers-metadata/per_year/2003.tsv: time stamp 2020-03-27 15:38:00.7760665 is 32629.658142062 s in the future
    tar: src/data/arxiv-papers-metadata/per_year/2004.tsv: time stamp 2020-03-27 15:38:00.8627758 is 32629.280954488 s in the future
    tar: src/data/arxiv-papers-metadata/per_year/2005.tsv: time stamp 2020-03-27 15:38:00.958988 is 32628.716523264 s in the future
    tar: src/data/arxiv-papers-metadata/per_year/2006.tsv: time stamp 2020-03-27 15:38:01.0722451 is 32628.327608349 s in the future
    tar: src/data/arxiv-papers-metadata/per_year/2007.tsv: time stamp 2020-03-27 15:38:01.2011785 is 32627.899425018 s in the future
    tar: src/data/arxiv-papers-metadata/per_year/2008.tsv: time stamp 2020-03-27 15:38:01.3199656 is 32627.433445145 s in the future
    tar: src/data/arxiv-papers-metadata/per_year/2009.tsv: time stamp 2020-03-27 15:38:01.4573527 is 32626.924672448 s in the future
    tar: src/data/arxiv-papers-metadata/per_year/2010.tsv: time stamp 2020-03-27 15:38:01.8662944 is 32626.600349585 s in the future
    tar: src/data/arxiv-papers-metadata/per_year/2011.tsv: time stamp 2020-03-27 15:38:02.0853306 is 32625.992426213 s in the future
    tar: src/data/arxiv-papers-metadata/per_year/2012.tsv: time stamp 2020-03-27 15:38:02.2559759 is 32625.258079226 s in the future
    tar: src/data/arxiv-papers-metadata/per_year/2013.tsv: time stamp 2020-03-27 15:38:02.4460549 is 32624.383405147 s in the future
    tar: src/data/arxiv-papers-metadata/per_year/2014.tsv: time stamp 2020-03-27 15:38:02.6586446 is 32623.528205943 s in the future
    tar: src/data/arxiv-papers-metadata/per_year/2015.tsv: time stamp 2020-03-27 15:38:02.9141423 is 32622.586554123 s in the future
    ^C


## 3) Test your model as a docker container


```python
# Remove previously deployed containers for this model
!docker rm -f research_predictor
```

    Error: No such container: reddit_predictor



```python
!docker run --name "research_predictor" -d --rm -p 5001:5000 research-classifier:0.1
```

    be29c6a00adec0f708dc5a1c83613e0656fddc06daba4ca02d93b5a7ece9b92b


### Make sure you wait for language model
SpaCy will download the English language model, so you have to make sure the container finishes downloading it before it can be used. You can view this by running the logs until you see "Linking successful".


```python
# Here we need to wait until we see "Linking successful", as it's downloading the Spacy English model
# You can hit stop when this happens
!docker logs -t -f research_predictor
```


```python
# We now test the REST endpoint expecting the same result
endpoint = "0.0.0.0:5001"
batch = sample
payload_type = "ndarray"

sc = SeldonClient(microservice_endpoint=endpoint)
response = sc.microservice(
    data=batch,
    method="predict",
    payload_type=payload_type,
    names=["tfidf"])

print(response)
```

    Success:True message:
    Request:
    data {
      names: "tfidf"
      ndarray {
        values {
          string_value: "This is the study that the article is based on:\r\n\r\nhttps://www.nature.com/articles/nature25778.epdf"
        }
      }
    }
    
    Response:
    meta {
    }
    data {
      names: "t:0"
      names: "t:1"
      ndarray {
        values {
          list_value {
            values {
              number_value: 0.8276709475641506
            }
            values {
              number_value: 0.1723290524358494
            }
          }
        }
      }
    }
    



```python
# We now stop it to run it in docker
!docker stop research_predictor
```

    reddit_predictor


## 4) Run Seldon in your kubernetes cluster


## Setup Seldon Core

Use the setup notebook to [Setup Cluster](../../seldon_core_setup.ipynb#Setup-Cluster) with [Ambassador Ingress](../../seldon_core_setup.ipynb#Ambassador) and [Install Seldon Core](../../seldon_core_setup.ipynb#Install-Seldon-Core). Instructions [also online](./seldon_core_setup.html).

## 5) Deploy your model with Seldon
We can now deploy our model by using the Seldon graph definition:


```python
%%writefile research-deployment.yaml
---
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: research-deployment
spec:
  name: research-spec
  predictors:
  - componentSpecs:
    - spec:
        containers:
        - image: research-classifier:0.1
          imagePullPolicy: IfNotPresent
          name: research-model
    graph:
      children: []
      name: research-model
      endpoint:
        type: REST
      type: MODEL
    name: research-pred
    replicas: 1

```

    Writing research-deployment.yaml



```python
!kubectl apply -f research-deployment.yaml
```

    seldondeployment.machinelearning.seldon.io/reddit-classifier created



```python
!kubectl get pods 
```

    NAME                                                    READY   STATUS    RESTARTS   AGE
    ambassador-7bfc87f865-jkxs8                             1/1     Running   0          5m2s
    ambassador-7bfc87f865-nr7bn                             1/1     Running   0          5m2s
    ambassador-7bfc87f865-q4lng                             1/1     Running   0          5m2s
    reddit-classifier-single-model-9199e4b-bcc5cdcc-g8j2q   2/2     Running   1          77s
    seldon-operator-controller-manager-0                    1/1     Running   1          5m23s


## 6) Interact with your model through API
Now that our Seldon Deployment is live, we are able to interact with it through its API.

There are two options in which we can interact with our new model. These are:

a) Using CURL from the CLI (or another rest client like Postman)

b) Using the Python SeldonClient

#### a) Using CURL from the CLI


```bash
%%bash
curl -X POST -H 'Content-Type: application/json' \
    -d "{'data': {'names': ['text'], 'ndarray': ['Hello world this is a test']}}" \
    http://127.0.0.1/seldon/default/research-deployment/api/v0.1/predictions
```

    {
      "meta": {
        "puid": "bvj1rjiq3vvnieo0oir4h7bf6f",
        "tags": {
        },
        "routing": {
        },
        "requestPath": {
          "classifier": "reddit-classifier:0.1"
        },
        "metrics": []
      },
      "data": {
        "names": ["t:0", "t:1"],
        "ndarray": [[0.6815614604065544, 0.3184385395934456]]
      }
    }

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100   372  100   300  100    72   1522    365 --:--:-- --:--:-- --:--:--  1897


#### b) Using the Python SeldonClient


```python
from seldon_core.seldon_client import SeldonClient
import numpy as np

host = "localhost"
port = "80" # Make sure you use the port above
batch = np.array(["Hello world this is a test"])
payload_type = "ndarray"
deployment_name="reddit-deployment"
transport="rest"
namespace="default"

sc = SeldonClient(
    gateway="ambassador", 
    ambassador_endpoint=host + ":" + port,
    namespace=namespace)

client_prediction = sc.predict(
    data=batch, 
    deployment_name=deployment_name,
    names=["text"],
    payload_type=payload_type,
    transport="rest")

print(client_prediction)
```

    Success:True message:
    Request:
    data {
      names: "text"
      ndarray {
        values {
          string_value: "Hello world this is a test"
        }
      }
    }
    
    Response:
    meta {
      puid: "uld2famhfrb97vd7regu0q7k32"
      requestPath {
        key: "classifier"
        value: "reddit-classifier:0.1"
      }
    }
    data {
      names: "t:0"
      names: "t:1"
      ndarray {
        values {
          list_value {
            values {
              number_value: 0.6815614604065544
            }
            values {
              number_value: 0.3184385395934456
            }
          }
        }
      }
    }
    



```python

```
