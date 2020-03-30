# Deployment & Explainability of Machine Learning COVID-19 Solutions at Scale with Seldon Core and Alibi

![](https://raw.githubusercontent.com/axsaucedo/seldon-core/corona_research_exploration/examples/models/research_paper_classification/diagram.jpg)

There has been great momentum from the machine learning community to extract insights from the increasingly growing COVID-19 Datasets, such as the Allen Institute for AI [Open Research Dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) as well as the data repository by [Johns Hopkins CSSE](https://github.com/CSSEGISandData/COVID-19).

We believe the most powerful insights can be brought through cross-functional collaboration, such as between infectious disease experts and machine learning practitioners. 

More importantly, once powerful predictive and explanatory models are built, it is especially important to be able to deploy and enable access to these models at scale to power solutions that can solve real-life challenges.

In this small tutorial we will show how you can deploy your machine learning solutions at scale, and we will use a practical example. For this we will be building a simple text classifier using the Allen Institute for AI COVID-19 Open Research Dataset which has been open sourced with over 44,000 scholarly articles on COVID-19, together with the [Arxiv Metadata Research Dataset](https://www.kaggle.com/tayorm/arxiv-papers-metadata) which contains over 1.5M papers.

In this tutorial we will focus primarily around the techniques to productionise an already trained model, and we will showcase how you're able to leverage the Seldon Core Prepackaged Model Servers, the Python Language Wrapper, and some of our AI Explainability infrastructure tools.

![](https://raw.githubusercontent.com/SeldonIO/seldon-core/master/doc/source/images/seldon-core-high-level.jpg)

## Tutorial Overview

The steps that we will be following in this tutorial include

1) Train and build a simple NLP model with SKLearn and SpaCy

2) Explain your model predictions using Alibi Explain

3) Containerize your model using Seldon Core Language Wrappers 

4) Deploy your model to Kubernetes

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

pd.set_option("display.notebook_repr_html", False)
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




                                                   abstract  is_covid
    3995  This article summarizes current knowledge abou...         1
    3996  While epidemiological models have traditionall...         1
    3997  TGEV and PEDV are porcine coronaviruses with t...         1
    3998  Metagenomics, i.e., the sequencing and analysi...         1
    3999  Population genetic diversity plays a prominent...         1



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




                   target
    precision    0.994898
    recall       0.975000
    specificity  0.995000
    accuracy     0.985000
    auc          0.985000
    f1           0.984848




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


## 3) Build your containerized model

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

### Test your model as a docker container


```python
# Remove previously deployed containers for this model
!docker rm -f research_predictor
```

    Error: No such container: research_predictor



```python
!docker run --name "research_predictor" -d --rm -p 5001:5000 research-classifier:0.1
```

    17d7c5cbe0bedd3bc5e6db9314ea27d4c901e50d8084531a3f7db96973a24a4b


### Make sure you wait for language model
SpaCy will download the English language model, so you have to make sure the container finishes downloading it before it can be used. You can view this by running the logs until you see "Linking successful".


```python
# Here we need to wait until we see "Linking successful", as it's downloading the Spacy English model
# You can hit stop when this happens
!docker logs -t research_predictor
```

    2020-03-27T06:42:15.300402971Z starting microservice
    2020-03-27T06:42:16.226289081Z 2020-03-27 06:42:16,225 - seldon_core.microservice:main:190 - INFO:  Starting microservice.py:main
    2020-03-27T06:42:16.227528306Z 2020-03-27 06:42:16,227 - seldon_core.microservice:main:246 - INFO:  Parse JAEGER_EXTRA_TAGS []
    2020-03-27T06:42:16.227623964Z 2020-03-27 06:42:16,227 - seldon_core.microservice:main:257 - INFO:  Annotations: {}
    2020-03-27T06:42:16.227703697Z 2020-03-27 06:42:16,227 - seldon_core.microservice:main:261 - INFO:  Importing ResearchClassifier
    2020-03-27T06:42:17.708431542Z Collecting en_core_web_sm==2.0.0 from https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz#egg=en_core_web_sm==2.0.0
    2020-03-27T06:42:18.766598177Z   Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz (37.4MB)
    2020-03-27T06:42:27.056778358Z Building wheels for collected packages: en-core-web-sm
    2020-03-27T06:42:27.057336058Z   Building wheel for en-core-web-sm (setup.py): started
    2020-03-27T06:42:29.508276329Z   Building wheel for en-core-web-sm (setup.py): finished with status 'done'
    2020-03-27T06:42:29.603323879Z   Created wheel for en-core-web-sm: filename=en_core_web_sm-2.0.0-cp37-none-any.whl size=37405978 sha256=f70bd20f4ab4f5557c58d319986f4742e89fa27e018546c20667b6680e2dacc5
    2020-03-27T06:42:29.603362554Z   Stored in directory: /tmp/pip-ephem-wheel-cache-qykoogkg/wheels/54/7c/d8/f86364af8fbba7258e14adae115f18dd2c91552406edc3fdaa
    2020-03-27T06:42:29.985450322Z Successfully built en-core-web-sm
    2020-03-27T06:42:29.985630047Z Installing collected packages: en-core-web-sm
    2020-03-27T06:42:30.064010513Z Successfully installed en-core-web-sm-2.0.0
    2020-03-27T06:42:31.268714990Z /opt/conda/lib/python3.7/site-packages/sklearn/base.py:318: UserWarning: Trying to unpickle estimator TfidfTransformer from version 0.20.1 when using version 0.22.2.post1. This might lead to breaking code or invalid results. Use at your own risk.
    2020-03-27T06:42:31.268758107Z   UserWarning)
    2020-03-27T06:42:31.268761465Z /opt/conda/lib/python3.7/site-packages/sklearn/base.py:318: UserWarning: Trying to unpickle estimator TfidfVectorizer from version 0.20.1 when using version 0.22.2.post1. This might lead to breaking code or invalid results. Use at your own risk.
    2020-03-27T06:42:31.268764174Z   UserWarning)
    2020-03-27T06:42:31.268766882Z /opt/conda/lib/python3.7/site-packages/sklearn/utils/deprecation.py:144: FutureWarning: The sklearn.linear_model.logistic module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.linear_model. Anything that cannot be imported from sklearn.linear_model is now part of the private API.
    2020-03-27T06:42:31.268772624Z   warnings.warn(message, FutureWarning)
    2020-03-27T06:42:31.268775115Z /opt/conda/lib/python3.7/site-packages/sklearn/base.py:318: UserWarning: Trying to unpickle estimator LogisticRegression from version 0.20.1 when using version 0.22.2.post1. This might lead to breaking code or invalid results. Use at your own risk.
    2020-03-27T06:42:31.268777824Z   UserWarning)
    2020-03-27T06:42:31.268792665Z 2020-03-27 06:42:31,268 - seldon_core.microservice:main:325 - INFO:  REST microservice running on port 5000
    2020-03-27T06:42:31.268795265Z 2020-03-27 06:42:31,268 - seldon_core.microservice:main:369 - INFO:  Starting servers
    2020-03-27T06:42:31.269566599Z 
    2020-03-27T06:42:31.269580140Z [93m    Linking successful[0m
    2020-03-27T06:42:31.269583282Z     /opt/conda/lib/python3.7/site-packages/en_core_web_sm -->
    2020-03-27T06:42:31.269585449Z     /opt/conda/lib/python3.7/site-packages/spacy/data/en_core_web_sm
    2020-03-27T06:42:31.269587507Z 
    2020-03-27T06:42:31.269589349Z     You can now load the model via spacy.load('en_core_web_sm')
    2020-03-27T06:42:31.269591515Z 
    2020-03-27T06:42:31.287201857Z  * Serving Flask app "seldon_core.wrapper" (lazy loading)
    2020-03-27T06:42:31.287236307Z  * Environment: production
    2020-03-27T06:42:31.287239340Z    WARNING: This is a development server. Do not use it in a production deployment.
    2020-03-27T06:42:31.287241724Z    Use a production WSGI server instead.
    2020-03-27T06:42:31.287243890Z  * Debug mode: off
    2020-03-27T06:42:31.287955315Z 2020-03-27 06:42:31,287 - werkzeug:_log:113 - INFO:   * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)


Now that it's running we can send a request to see the output. We will see that the response is basically the output of the model that we just trained, but it's now exposed through a fully fledged REST API.


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
          string_value: "The modifiable areal unit problem, MAUP, is ever-present although not always appreciated. Through real examples, this article outlines the basic causes of MAUP, namely changes in the size, shape, and/or orientation of spatial categories/polygons used to map areal data. The visual effects of changes to mapped data are obvious even though the impacts on our understanding of the world are profound. The article concludes with a discussion of technical and broader strategic approaches for confronting the effects of MAUP on our treatment and interpretation of areal data."
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
              number_value: 0.550335993635677
            }
            values {
              number_value: 0.449664006364323
            }
          }
        }
      }
    }
    


Before we move to the next step and run our model in a Kubernetes cluster we can stop the docker container.


```python
# We now stop it to run it in docker
!docker stop research_predictor
```

    research_predictor


## 4) Run Seldon in your kubernetes cluster

In order to deploy our model to Kubernetes, we just need to define a simple deployment configuration file.

This configuration file basically points to the container that we just built above, so make sure the Kubernetes cluster has access to the container (you may need to push the image into your docker repo).


```python
%%writefile research-deployment.yaml
---
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: research-deployment
spec:
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
    name: default
    replicas: 1

```

    Overwriting research-deployment.yaml


Now we can apply this SeldonDeployment into our Kubernetes cluster


```python
!kubectl apply -f research-deployment.yaml
```

    seldondeployment.machinelearning.seldon.io/research-deployment created


And we can make sure that our model is actually running as expected


```python
!kubectl get pods | grep research
```

    research-deployment-research-pred-0-65f7646d9c-s6bnr              2/2     Running     0          54s


## 5) Send requests to our deployed model

Now that our Seldon Deployment is live, we are able to interact with it through its API.

There are two options in which we can interact with our new model. These are:

a) Using CURL from the CLI (or another rest client like Postman)

b) Using the Python SeldonClient

#### a) Using CURL from the CLI

We can actually send a simple request and see what the prediction is, in the case below the prediction is positive for COVID


```bash
%%bash
curl -X POST -H 'Content-Type: application/json' \
    -d '{"data": {"names": ["text"], "ndarray": ["This paper is about virus and spread of disease"]}}' \
    http://localhost/seldon/default/research-deployment/api/v1.0/predictions
```

    {"data":{"names":["t:0","t:1"],"ndarray":[[0.3729505481093134,0.6270494518906866]]},"meta":{}}


      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100   188  100    95  100    93   7916   7750 --:--:-- --:--:-- --:--:-- 15666


#### b) Using the Python SeldonClient


```python
from seldon_core.seldon_client import SeldonClient
import numpy as np

host = "localhost"
port = "80" # Make sure you use the port above
batch = np.array(["This paper is about virus and spread of disease"])
payload_type = "ndarray"
deployment_name="research-deployment"
transport="rest"
namespace="default"

sc = SeldonClient(
    gateway="ambassador", 
    gateway_endpoint=host + ":" + port,
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
    meta {
    }
    data {
      names: "text"
      ndarray {
        values {
          string_value: "This paper is about virus and spread of disease"
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
              number_value: 0.3729505481093134
            }
            values {
              number_value: 0.6270494518906866
            }
          }
        }
      }
    }
    


## 6) Deploy a text explainer

Now that we have deployed our model, we can also deploy the explainer that we showed above.

Fortunately, Seldon Core already supports several out of the box explainers, including a generic text explainer.

Because of this, we will be able to just modify the existing deployment file to specify we want to add a text explainer


```python
%%writefile research-deployment-with-explainer.yaml
---
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: research-deployment
spec:
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
    explainer:
      type: AnchorText
    name: default
    replicas: 1
```

    Overwriting research-deployment-with-explainer.yaml


As you can see we defined a new section with the name `explainer:`, and we provided the type as `AnchorText`.

This will ensure that we deploy our model together with an explainer of type AnchorText which is what we used above.


```python
!kubectl apply -f research-deployment-with-explainer.yaml
```

    seldondeployment.machinelearning.seldon.io/research-deployment created


We can now wait until our new predictor explainer is deployed, and all the routes are created


```python
!kubectl get pods | grep research
```

    research-deployment-default-0-research-model-86f486b5bd-x6vq5   2/2     Running     0          14m
    research-deployment-default-explainer-5b87f8c744-6bj7q          1/1     Running     1          14m


## 7) Send requests to our deployed explainer

Now that our Explainer is live, we are able to interact with it through its API.

The explainer will be interacting with the model to reverse engineer the predictions.


```bash
%%bash
curl -X POST -H 'Content-Type: application/json' \
    -d '{"data": {"names": ["text"], "ndarray": ["This paper is about virus and spread of disease"]}}' \
    http://localhost:80/seldon/default/research-deployment/default/explainer/api/v1.0/explain | json_pp
```

    {
       "coverage" : 0.4993,
       "meta" : {
          "name" : "AnchorText"
       },
       "precision" : 1,
       "raw" : {
          "num_preds" : 1000001,
          "positions" : [
             40
          ],
          "names" : [
             "disease"
          ],
          "precision" : [
             1
          ],
          "prediction" : 1,
          "examples" : [
             {
                "covered_true" : [
                   [
                      "UNK paper is about virus UNK spread UNK disease"
                   ],
                   [
                      "UNK paper UNK about UNK UNK UNK UNK disease"
                   ],
                   [
                      "This paper is UNK UNK and UNK UNK disease"
                   ],
                   [
                      "UNK paper is UNK UNK and UNK UNK disease"
                   ],
                   [
                      "UNK paper is about UNK UNK spread UNK disease"
                   ],
                   [
                      "UNK UNK is UNK UNK UNK spread of disease"
                   ],
                   [
                      "This paper UNK UNK UNK and UNK UNK disease"
                   ],
                   [
                      "UNK UNK is about UNK UNK UNK UNK disease"
                   ],
                   [
                      "This UNK is about virus and spread UNK disease"
                   ],
                   [
                      "UNK UNK UNK about UNK UNK spread of disease"
                   ]
                ],
                "uncovered_false" : [],
                "covered" : [
                   [
                      "This paper UNK about UNK UNK spread UNK disease"
                   ],
                   [
                      "UNK UNK is UNK UNK UNK spread of disease"
                   ],
                   [
                      "UNK UNK UNK UNK UNK UNK spread UNK disease"
                   ],
                   [
                      "UNK paper UNK about virus UNK UNK of disease"
                   ],
                   [
                      "UNK paper UNK about virus and UNK of disease"
                   ],
                   [
                      "UNK UNK is about UNK UNK UNK UNK disease"
                   ],
                   [
                      "This paper UNK about UNK UNK spread of disease"
                   ],
                   [
                      "This UNK is UNK UNK UNK spread UNK disease"
                   ],
                   [
                      "This paper is about virus and spread UNK disease"
                   ],
                   [
                      "This paper is about UNK and spread UNK disease"
                   ]
                ],
                "uncovered_true" : [],
                "covered_false" : []
             }
          ],
          "instance" : "This paper is about virus and spread of disease",
          "coverage" : [
             0.4993
          ],
          "mean" : [
             1
          ],
          "feature" : [
             8
          ],
          "all_precision" : 0
       },
       "names" : [
          "disease"
       ]
    }


      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  1529  100  1436  100    93   1447     93  0:00:01 --:--:--  0:00:01  1539



```python

```
