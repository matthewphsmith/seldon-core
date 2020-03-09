# Payload Logging

Logging of request and response payloads from your Seldon Deployment can be accomplished by adding a logging section to any part of the Seldon deployment graph. An example is shown below:

```
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: seldon-model
spec:
  name: test-deployment
  predictors:
  - componentSpecs:
    - spec:
        containers:
        - image: seldonio/mock_classifier:1.3
          name: classifier
    graph:
      children: []
      endpoint:
        type: REST
      name: classifier
      type: MODEL
      logger:
        url: http://mylogging-endpoint
        mode: all
    name: example
    replicas: 1

```

The logging for the top level requets response is provided by:

```
      logger:
        url: http://mylogging-endpoint
        mode: all
```

In this example both request and response payloads as specified by the `mode` attribute are sent as CloudEvents to the url `http://mylogging-endpoint`.

The specification is:

 * url: Any url. Optional. If not provided then it will default to the default knative borker in the namespace of the Seldon Deployment.
 * mode: Either `request`, `response` or `all`

## Global Request Logger Options

In order to avoid having to configure the request logger url every time, you can set a default URL that will apply to all your deployments. This can be done through the helm chart variable:

`values.yaml`:
```yaml
...
executor:
  defaultRequestLoggerEndpointPrefix: 'http://default-broker.'
...
```

The custom prefix provided in this case will always be suffixed by the namespace in which the deployment runs.

It can still be overriden through the SeldonDeployment values, when provided as part of the deployment file, as these will take precedence.

## Example Notebook

You can try out an [example notebook with logging](../examples/payload_logging.html)

