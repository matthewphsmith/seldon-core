# Seldon Complex Stream Processing

This example will walk you throu
gh the steps for you to be able to create a model and deploy it as a streaming service, as Seldon supports REST, GRPC and KAFKA servers. 

In this example we will:
1) Create a simple model wrapper
2) Containerise the model
3) Test the model locally
4) Deploy the model and test in a kubernetes cluster

## 1-3) Create your model

This is a continuation of our first tutorial showing you how to create a single streaming model.

Please complete that example before you proceed to this example.

Once you have successfully run your first single-streaming-model we'll be able to see how to build complex streaming graphs with Seldon.


## 4) Deploy the model and test in a Kubernetes cluster

We now want to test it in our Kubernetes cluster. 

For this, you will need to make sure you have all the [Seldon Core dependencies installed (Operator, Ingress, etc).](https://docs.seldon.io/projects/seldon-core/en/latest/workflow/install.html)

Once you have everything installed, we'll do the following steps:
1) Run Kafka in our Kubernetes
2) Create a Seldon Deployment that uses our model
3) Deploy our Seldon Deployment
4) Publish messages in the input topic and see messages coming from output topic
5) Run a benchmark for performance

### 4.1) Run Kafka in our Kubernetes

We first need to make sure our helm installer has access to the incubator charts:

```console
helm repo add bitnami https://charts.bitnami.com/bitnami 
```

Now we're able to create a simple Kafka deployment:

```console
helm install my-kafka bitnami/kafka
```

Once it's running we'll be able to see the containers:

```console
$ kubeclt get pods

NAME                   READY   STATUS    RESTARTS   AGE
my-kafka-0             1/1     Running   0          2m43s
my-kafka-1             1/1     Running   0          42s
my-kafka-zookeeper-0   1/1     Running   0          2m43s
my-kafka-zookeeper-1   1/1     Running   0          96s
my-kafka-zookeeper-2   1/1     Running   0          62s
```

### 4.2) Create a Seldon Deployment that uses our model

Now we want to create a Seldon Deploymen configuration file that we'll be able to deploy.

For this we'll use a simple file that just sets the following:
* Selects the deployment to run without the Engine/Orchestrator
* Adds the environment variables to point it to the cluster we just deployed
* Points to the docker image that we just built

The contents of `cluster/streaming_model_deployment.json` are as follows:

```json
{    "apiVersion": "machinelearning.seldon.io/v1alpha2"
,
    "kind": "SeldonDeployment",
    "metadata": {
        "name": "streaming-deployment",
        "creationTimestamp": null
    },
    "spec": {
        "name": "streaming-spec",
        "predictors": [
            {
                "name": "streaming-graph",
                "graph": {
                    "name": "streaming-model-one",
                    "endpoint": {
                        "type": "REST",
                        "service_port": 9000
                    },
                    "type": "MODEL",
                    "children": [
                        {
                            "name": "streaming-model-two",
                            "endpoint": {
                                "type": "REST",
                                "service_port": 9000
                            },
                            "type": "MODEL",
                            "children": [
                                {
                                    "name": "streaming-model-three",
                                    "endpoint": {
                                        "type": "REST",
                                        "service_port": 9000
                                    },
                                    "type": "MODEL",
                                    "children": []
                                }
                            ]
                        }
                    ],
                    "parameters": []
                },
                "componentSpecs": [
                    {
                        "spec": {
                            "containers": [
                                {
                                    "image": "streaming_model:0.1",
                                    "name": "streaming-model-one",
                                    "env": [
                                        {

                                            "name": "PREDICTIVE_UNIT_STREAMING_BROKER",
                                            "value": "kafka://my-kafka.default.svc.cluster.local:9092"
                                        }
                                    ]
                                },
                                {
                                    "image": "streaming_model:0.1",
                                    "name": "streaming-model-two",
                                    "env": [
                                        {

                                            "name": "PREDICTIVE_UNIT_STREAMING_BROKER",
                                            "value": "kafka://my-kafka.default.svc.cluster.local:9092"
                                        }
                                    ]
                                },
                                {
                                    "image": "streaming_model:0.1",
                                    "name": "streaming-model-three",
                                    "env": [
                                        {

                                            "name": "PREDICTIVE_UNIT_STREAMING_BROKER",
                                            "value": "kafka://my-kafka.default.svc.cluster.local:9092"
                                        }
                                    ]
                                }
                            ],
                            "terminationGracePeriodSeconds": 1
                        }
                    }
                ],
                "replicas": 1,
                "engineResources": {},
                "svcOrchSpec": {},
                "traffic": 100,
                "explainer": {
                    "containerSpec": {
                        "name": "",
                        "resources": {}
                    }
                }
            }
        ],
        "annotations": {
            "seldon.io/engine-seldon-log-messages-externally": "true"
        }
    },
    "status": {}
}
```


### 4.3) Deploy our Seldon Deployment

Now that we've created out deployment, we just need to launch it:

```console
kubectl apply -f cluster/streaming_model_deployment.json
```

Once it's deployed we can see it by running:

```console
$ kubectl get pods | grep streaming

streaming-spec-streaming-graph-e90bdcd-56986c5d4b-7xvtm   1/1     Running   0          6m28s
```

### 4.4) Publish messages in the input topic and see messages coming from output topic

Now we want to test it by sending some messages.

We can get the name of the pod by running:

```console
export STREAM_SELDON_POD=`kubectl get pod -l seldon-app=streaming-deployment-streaming-spec-streaming-graph -o jsonpath="{.items[0].metadata.name}"`
```

First let's run a consumer to see the output:

```python
kubectl exec -i $STREAM_SELDON_POD -c streaming-model-one python - <<EOF
import kafka
consumer = kafka.KafkaConsumer(
    'streaming-deployment-output',
    bootstrap_servers='my-kafka.default.svc.cluster.local:9092');
print(next(consumer).value)
EOF
```

Then let's send the message:

```python
kubectl exec -i $STREAM_SELDON_POD -c streaming-model-one python - <<EOF
import kafka, json;
producer = kafka.KafkaProducer(
    bootstrap_servers='my-kafka.default.svc.cluster.local:9092',
    key_serializer=lambda v: json.dumps(v).encode('utf-8'),
    value_serializer=lambda v: json.dumps(v).encode('utf-8'));
result = producer.send('streaming-deployment-input',key={"id": 1}, value={'data': { 'ndarray': [1,2,3,4] } })
result.get(timeout=3)
EOF
```

We can now see in our consumer that we have received and printed the output as follows:

```json
b'{"data": {"ndarray": [1, 2, 3, 4], "names": []}, "meta": {}}'
```

## 4.5) Benchmark

It's possible to run benchmarks to understand the performance of these models.

To do this, we will be able to run a set of commands provided out of the box by kafka.

We will be able to leverage these commands by running them directly into our Kafka container.

THe first command that we want to run is a consumer perfomance tester:

```
kubectl exec -i my-kafka-0 -- /opt/bitnami/kafka/bin/kafka-consumer-perf-test.sh --topic streaming-deployment-output --messages 50000 --broker-list my-kafka.default.svc.cluster.local:9092 --timeout 100000 --show-detailed-stats --from-latest
```

ONce this is running we now can run our command to actually send some data.

In order to run the following command you will have to create a file with some payload, so the easiest way is to connect to the node and run the commands.

You can connect to the node running the following command:

```
kubectl exec -it my-kafka-0 bash
```

Once you are connected, you want to createa payload file as follows:

```
echo '{"data":{"ndarray":[1,2,3,4]}}' > payload.json
```

Finally you can send the test data running the following command:

```
kafka-run-class.sh org.apache.kafka.tools.ProducerPerformance --topic streaming-deployment-input --num-records 50 --payload-file payload.json --throughput -1 --producer-props acks=1 bootstrap.servers=my-kafka.default.svc.cluster.local:9092 buffer.memory=67108864 batch.size=8196"
```

The data should be sent pretty quick, and you will start seeing performance metrics in the consumer script you should have started above.

You can also see the logs of the container to make sure that everything is running as expected

