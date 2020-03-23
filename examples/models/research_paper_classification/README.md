# Research Paper Classification for COVID-19 Research

There has been great momentum from the machine learning community to extract insights from the increasingly growing COVID-19 Datasets.

We believe the most powerful insights can be brought through cross-functional collaboration between infectious disease experts and machine learning practitioners. 

More importantly, once powerful predictive and explanatory models are built, it is especially important to be able to enable access to these models at scale.

In this small tutorial we will show how you can deploy your machine learning solutions at scale, and we will use a practical example with an on-going Kaggle dataset released by the Allen Institute for AI containing over 44,000 scholarly articles on COVID-19.

In this tutorial we will deploy a COVID-19 research paper classifier using Seldon Core, which will allow us to convert this ML model into a fully fledged microservice, which we'll be able to send REST / GRPC requests, as well as monitor through grafana / ELK integration.

## Tutorial Overview

In this tutorial we will showcase an end-to-end workflow that will ultimately show how to deploy a machine learning model - in this case we will be building a classifier that identifies whether a research paper is related to covid-19.

The steps in this tutorial include:

1) Train and build your NLP model with SKLearn and SpaCy

2) Explain your model predictions using Alibi Explain

3) Containerize your model using Seldon Core Language Wrappers and deploy to Kubernetes

5) Test your deployed model by sending requests

6) Deploy our standard Alibi TextExplainer 

7) Test your deployed explainer by sending requests

### Before you start
Make sure you install the following dependencies, as they are critical for this example to work:

* Seldon Core v1.1+ installed with Istio Ingress Enabled ([Documentation Instructions](https://docs.seldon.io/projects/seldon-core/en/latest/workflow/install.html#ingress-support))
* All dependencies specified in the Seldon Core page

Let's get started! 🚀🔥

## 0) Prepare your environment

First we want to install all the dependencies. For this let's create a requirements-dev.txt file with everything we'll need:


```python
%%writefile requirements-dev.txt
scipy>= 0.13.3
scikit-learn>=0.18
spacy==2.0.18
dill==0.2.9
xai==0.0.5
alibi==0.4.0
```

    Writing requirements-dev.txt


And then let's install all of our dependencies locally so we can train and test our model


```python
# Let's first install any dependencies
!pip install -r requirements-dev.txt
```

Now that everything is installed, we can import all our dependencies


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

# This import may take a while as it will download the Spacy ENGLISH model
from ml_utils import CleanTextTransformer, SpacyTokenTransformer
```

    
    [93m    Linking successful[0m
        /home/alejandro/miniconda3/lib/python3.7/site-packages/en_core_web_sm
        -->
        /home/alejandro/miniconda3/lib/python3.7/site-packages/spacy/data/en_core_web_sm
    
        You can now load the model via spacy.load('en_core_web_sm')
    


## 1) Train and build your NLP model with SKLearn and SpaCy

We can now get started with the training of our model. 

For this we will want to load the simplified dataset that we have created for this example:


```python
df = pd.read_csv("./data/research_paper_abstracts.csv")
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




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc691cd95d0>




![png](README_files/README_12_1.png)


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


```python
# Clean the text
clean_text_transformer = CleanTextTransformer()
x_train_clean = clean_text_transformer.transform(x_train)
```

### Train our model: Tokenize
We now convert our input text into tokens - for this we use the SpaCy module.


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
            preprocessor=<function <lambda> at 0x7fc69197e3b0>,
            smooth_idf=True, stop_words=None, strip_accents=None,
            sublinear_tf=False, token_pattern=None,
            tokenizer=<function <lambda> at 0x7fc69197e560>, use_idf=True,
            vocabulary=None)




```python
# Transform our tokens to tfidf vectors
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



### Test your model

Now that we've trained our model we can test its performance against our test dataset. 

FOr this we first extract the predictions for all our test dataset


```python
def predict_fn(x):
    x_c = clean_text_transformer.transform(x)
    x_s = spacy_tokenizer.transform(x_c)
    x_t = tfidf_vectorizer.transform(x_s)
    return lr.predict(x_t)
pred = predict_fn(x_test)
```

And now we can see the performance of the predictions, which as we can see is quite satisfactory


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




![png](README_files/README_27_1.png)


## 2) Explain your model predictions using Alibi Explain

We will now use the Alibi library to explain predictions from the model we've built


```python
# Import the Spacy NLP module for our explainer
from ml_utils import nlp
```


```python
explainer = alibi.explainers.AnchorText(nlp, predict_fn)
```


```python
x_explain = x_test[1]
x_explain
```




    'The modifiable areal unit problem, MAUP, is ever-present although not always appreciated. Through real examples, this article outlines the basic causes of MAUP, namely changes in the size, shape, and/or orientation of spatial categories/polygons used to map areal data. The visual effects of changes to mapped data are obvious even though the impacts on our understanding of the world are profound. The article concludes with a discussion of technical and broader strategic approaches for confronting the effects of MAUP on our treatment and interpretation of areal data.'




```python
explanation = explainer.explain(x_explain, threshold=0.95, use_unk=True)
```


```python
print('Anchor: %s' % (' AND '.join(explanation.anchor)))
print('Precision: %.2f' % explanation.precision)
print(f"\nOriginal Sample:\n{x_explain}")
print('\nFirst Example where anchor applies and model predicts is_covid==True')
print(f"\n{explanation.raw['examples'][-1]['covered_true'][0]}".replace("UNK", "___"))
print('\n\nExample where anchor applies and model predicts is_covid==False')
print(f"\n{explanation.raw['examples'][-1]['covered_false'][0]}".replace("UNK", "___"))
```

    Anchor: the
    Precision: 0.98
    
    Original Sample:
    The modifiable areal unit problem, MAUP, is ever-present although not always appreciated. Through real examples, this article outlines the basic causes of MAUP, namely changes in the size, shape, and/or orientation of spatial categories/polygons used to map areal data. The visual effects of changes to mapped data are obvious even though the impacts on our understanding of the world are profound. The article concludes with a discussion of technical and broader strategic approaches for confronting the effects of MAUP on our treatment and interpretation of areal data.
    
    First Example where anchor applies and model predicts is_covid==True
    
    The modifiable ___ unit problem , MAUP ___ ___ ever - ___ ___ not always ___ . ___ ___ ___ , this ___ ___ the basic ___ of ___ , ___ ___ in the ___ ___ ___ ___ and/or ___ of spatial categories ___ polygons ___ ___ map ___ data ___ The visual effects of ___ ___ mapped ___ are ___ ___ though the impacts on ___ understanding ___ ___ world ___ profound . ___ article concludes with ___ ___ ___ technical and broader strategic ___ ___ ___ the ___ ___ ___ on our ___ and ___ of areal ___ ___
    
    
    Example where anchor applies and model predicts is_covid==False
    
    ___ modifiable ___ ___ problem ___ MAUP , is ___ ___ present although not ___ ___ ___ ___ ___ ___ , ___ article outlines the basic causes of MAUP , ___ changes ___ the size ___ ___ , and/or orientation ___ spatial categories ___ polygons used ___ ___ areal ___ ___ The ___ ___ of changes to ___ ___ ___ ___ even though ___ ___ ___ ___ understanding of ___ world are profound . ___ article concludes with a ___ ___ technical and broader ___ approaches ___ confronting the effects ___ MAUP ___ ___ treatment ___ ___ of areal ___ ___


First we need to export the trained models

## 2) Build your containerized model


```python
# These are the models we'll deploy
with open('tfidf_vectorizer.model', 'wb') as model_file:
    dill.dump(tfidf_vectorizer, model_file)
with open('lr.model', 'wb') as model_file:
    dill.dump(lr, model_file)
```

Now we write a class wrapper to expose the models


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



```python
# test that our model works
from ResearchClassifier import ResearchClassifier
# With one sample
sample = x_test[1:2]
print(sample)
print(ResearchClassifier().predict(sample, ["feature_name"]))
print(y_test[1:2])
```

    ['The modifiable areal unit problem, MAUP, is ever-present although not always appreciated. Through real examples, this article outlines the basic causes of MAUP, namely changes in the size, shape, and/or orientation of spatial categories/polygons used to map areal data. The visual effects of changes to mapped data are obvious even though the impacts on our understanding of the world are profound. The article concludes with a discussion of technical and broader strategic approaches for confronting the effects of MAUP on our treatment and interpretation of areal data.']
    [[0.5503458 0.4496542]]
    [1]


### Create Docker Image with the S2i utility
Using the S2I command line interface we wrap our current model to seve it through the Seldon interface


```python
# To create a docker image we need to create the .s2i folder configuration as below:
!cat .s2i/environment
```

    MODEL_NAME=RedditClassifier
    API_TYPE=REST
    SERVICE_TYPE=MODEL
    PERSISTENCE=0



```python
%%writefile requirements.txt
scipy>= 0.13.3
scikit-learn>=0.18
spacy==2.0.18
dill==0.2.9
```

    Overwriting requirements.txt



```python
!s2i build . seldonio/seldon-core-s2i-python3:0.18 research-classifier:0.1
```

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
