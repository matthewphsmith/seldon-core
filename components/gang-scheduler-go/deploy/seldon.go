package deploy

import (
	sc "github.com/seldonio/seldon-core/operator/apis/machinelearning/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	typesv1 "k8s.io/apimachinery/pkg/types"
)

const (
	seldonAPIVersion    = "machinelearning.seldon.io/v1"
	seldonKind          = "SeldonDeployment"
	seldonMainPredictor = "main"
	seldonModel         = "model"
)

type SeldonDeployment struct {
	Deployment `json:"-"`
	sc.SeldonDeployment
}

// NewSeldonDeploymentWithDetails creates a new SeldonDeployment spec from
// scratch based on the given deployment details.
func NewSeldonDeploymentWithDetails(det *DeploymentDetails) Deployment {
	sdep := &SeldonDeployment{
		SeldonDeployment: sc.SeldonDeployment{
			TypeMeta: metav1.TypeMeta{
				APIVersion: seldonAPIVersion,
				Kind:       seldonKind,
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      det.Name,
				Namespace: det.Namespace,
			},
		},
	}

	pred := newSeldonPredictor(det)

	sdep.Spec = sc.SeldonDeploymentSpec{
		Name:       det.Name,
		Predictors: []sc.PredictorSpec{*pred},
	}

	return sdep
}

func newSeldonPredictor(det *DeploymentDetails) *sc.PredictorSpec {
	modelType := sc.MODEL

	pred := &sc.PredictorSpec{
		Name: seldonMainPredictor,
		ComponentSpecs: []*sc.SeldonPodSpec{
			{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  seldonModel,
							Image: det.ModelImage,
							Env:   det.ModelEnv,
						},
					},
				},
			},
		},
		Graph: &sc.PredictiveUnit{
			Name: seldonModel,
			Type: &modelType,
		},
	}

	// nolint:gomnd
	if det.Replicas > 0 {
		pred.Replicas = det.Replicas
	}

	return pred
}

// Details returns the common set of deployment information (e.g. name, namespace,
// etc.)
func (sd *SeldonDeployment) Details() *DeploymentDetails {
	det := &DeploymentDetails{
		Kind:      sd.Kind,
		Namespace: sd.Namespace,
		Name:      sd.Name,
	}

	pred := sd.getMainPredictor()
	if pred != nil {
		det.Replicas = pred.Replicas
	}

	modelContainer := sd.getModelContainer(pred)
	if modelContainer != nil {
		det.ModelImage = modelContainer.Image
		det.ModelEnv = modelContainer.Env
	}

	return det
}

func (sd *SeldonDeployment) getMainPredictor() *sc.PredictorSpec {
	predictors := sd.Spec.Predictors
	if len(predictors) == 0 {
		return nil
	}

	// TODO: Improve this logic to detect which one is canary / shadow.
	return &predictors[0]
}

func (sd *SeldonDeployment) getModelContainer(pred *sc.PredictorSpec) *v1.Container {
	if pred == nil {
		return nil
	}

	specs := pred.ComponentSpecs
	if len(specs) == 0 {
		return nil
	}

	podSpec := specs[0].Spec

	containers := podSpec.Containers
	if len(containers) == 0 {
		return nil
	}

	// TODO: Improve this logic to find model's spec matching by name
	return &containers[0]
}

func (sd *SeldonDeployment) WithDetails(det *DeploymentDetails) (Deployment, error) {
	deepCopy := sd.SeldonDeployment.DeepCopy()
	withDetails := &SeldonDeployment{SeldonDeployment: *deepCopy}

	// Can Name and Kind be considered immutable?
	if det.Namespace != "" {
		withDetails.Namespace = det.Namespace
	}

	if det.ModelImage != "" {
		pred := withDetails.getMainPredictor()
		if pred == nil {
			return nil, deploymentDetailsError(sd, "No predictors were found")
		}

		if det.Replicas > 0 {
			pred.Replicas = det.Replicas
		}

		// TODO: Should we consider creating the container spec if none exists?
		modelContainer := withDetails.getModelContainer(pred)
		if modelContainer == nil {
			return nil, deploymentDetailsError(sd, "Model container wasn't found")
		}

		modelContainer.Image = det.ModelImage
		modelContainer.Env = det.ModelEnv
	}

	return withDetails, nil
}

func (sd *SeldonDeployment) WithoutStatus() Deployment {
	deepCopy := sd.SeldonDeployment.DeepCopy()
	wo := &SeldonDeployment{SeldonDeployment: *deepCopy}

	wo.SetUID(typesv1.UID(""))
	wo.SetSelfLink("")
	wo.SetGeneration(0)
	wo.SetResourceVersion("")
	annotations := wo.GetAnnotations()
	delete(annotations, "kubectl.kubernetes.io/last-applied-configuration")
	wo.SetAnnotations(annotations)
	wo.SetCreationTimestamp(metav1.Time{})
	wo.Status = sc.SeldonDeploymentStatus{}

	return wo
}
