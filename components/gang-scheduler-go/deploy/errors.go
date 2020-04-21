package deploy

import (
	"fmt"
	"io/ioutil"
	"net/http"
)

func namespaceNotFoundError(namespace string) error {
	return fmt.Errorf("Namespace %s not found", namespace)
}

func deploymentNotFoundError(namespace string, name string) error {
	return fmt.Errorf("Deployment %s on namespace %s not found", name, namespace)
}

func deploymentDetailsError(deployment Deployment, reason string) error {
	det := deployment.Details()
	return fmt.Errorf("Deployment %s couldn't be updated: %s", det.Name, reason)
}

func checkResponseError(res *http.Response) error {
	if res.StatusCode == http.StatusOK {
		return nil
	}

	mes, err := ioutil.ReadAll(res.Body)
	if err != nil {
		return err
	}

	return fmt.Errorf(string(mes))
}

func authError() error {
	return fmt.Errorf("unexpected error authenticating")
}
