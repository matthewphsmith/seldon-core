package deploy

import (
	"encoding/json"
	"io/ioutil"

	v1 "k8s.io/api/core/v1"
)

// DeploymentDetails represents a small subset of data between KFServing and
// Seldon Core. It enables us to have a common interface to interact with both.
type DeploymentDetails struct {
	Name       string
	Kind       string
	Namespace  string
	ModelImage string
	ModelEnv   []v1.EnvVar
	Replicas   int32
}

type Deployment interface {
	Details() *DeploymentDetails
	WithDetails(*DeploymentDetails) (Deployment, error)
	WithoutStatus() Deployment
}

// NewDeployment will read a filepath and will create a Deployment with the
// right type.
func NewDeployment(filepath string) (Deployment, error) {
	file, err := ioutil.ReadFile(filepath)
	if err != nil {
		return nil, err
	}

	// TODO: Add support for InferenceService deployments.
	var sdep *SeldonDeployment

	err = json.Unmarshal(file, &sdep)
	if err != nil {
		return nil, err
	}

	return sdep, nil
}

func NewDeploymentFromDetails(det *DeploymentDetails) Deployment {
	// TODO: Add support for InferenceService deployments.
	return NewSeldonDeploymentWithDetails(det)
}
