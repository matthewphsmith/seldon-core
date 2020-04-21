package deploy

import (
	"testing"

	. "github.com/onsi/gomega"
	sc "github.com/seldonio/seldon-core/operator/apis/machinelearning/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestNewSeldonPredictor(t *testing.T) {
	g := NewGomegaWithT(t)

	modelType := sc.MODEL
	tests := []struct {
		det  *DeploymentDetails
		pred *sc.PredictorSpec
	}{
		{
			det: &DeploymentDetails{
				Name:       "my-model",
				ModelImage: "seldonio/fixed-model:v1.0",
				ModelEnv: []v1.EnvVar{
					{
						Name:  "MODEL_WEIGHTS",
						Value: "gs://test/model-1.json",
					},
				},
				Replicas: 4,
			},
			pred: &sc.PredictorSpec{
				Name:     seldonMainPredictor,
				Replicas: 4,
				ComponentSpecs: []*sc.SeldonPodSpec{
					{
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Name:  seldonModel,
									Image: "seldonio/fixed-model:v1.0",
									Env: []v1.EnvVar{
										{
											Name:  "MODEL_WEIGHTS",
											Value: "gs://test/model-1.json",
										},
									},
								},
							},
						},
					},
				},
				Graph: &sc.PredictiveUnit{
					Name: seldonModel,
					Type: &modelType,
				},
			},
		},
	}

	for _, test := range tests {
		pred := newSeldonPredictor(test.det)

		g.Expect(pred).To(Equal(test.pred))
	}
}

// nolint:funlen
func TestNewSeldonDeploymentWithDetails(t *testing.T) {
	g := NewGomegaWithT(t)

	modelType := sc.MODEL
	tests := []struct {
		det  *DeploymentDetails
		sdep *SeldonDeployment
	}{
		{
			det: &DeploymentDetails{
				Kind:       "SeldonDeployment",
				Namespace:  "staging",
				Name:       "my-model",
				ModelImage: "seldonio/fixed-model:v1.0",
			},
			sdep: &SeldonDeployment{
				SeldonDeployment: sc.SeldonDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "staging",
						Name:      "my-model",
					},
					TypeMeta: metav1.TypeMeta{
						APIVersion: seldonAPIVersion,
						Kind:       seldonKind,
					},
					Spec: sc.SeldonDeploymentSpec{
						Name: "my-model",
						Predictors: []sc.PredictorSpec{
							{
								Name: seldonMainPredictor,
								ComponentSpecs: []*sc.SeldonPodSpec{
									{
										Spec: v1.PodSpec{
											Containers: []v1.Container{
												{
													Name:  seldonModel,
													Image: "seldonio/fixed-model:v1.0",
												},
											},
										},
									},
								},
								Graph: &sc.PredictiveUnit{
									Name: seldonModel,
									Type: &modelType,
								},
							},
						},
					},
				},
			},
		},
	}

	for _, test := range tests {
		sdep := NewSeldonDeploymentWithDetails(test.det)

		g.Expect(sdep).To(Equal(test.sdep))
	}
}

// nolint:funlen
func TestSeldonDeploymentDetails(t *testing.T) {
	g := NewGomegaWithT(t)

	tests := []struct {
		sdep     *SeldonDeployment
		expected *DeploymentDetails
	}{
		{
			sdep: &SeldonDeployment{
				SeldonDeployment: sc.SeldonDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "staging",
						Name:      "sdep-no-spec",
					},
					TypeMeta: metav1.TypeMeta{
						Kind: "SeldonDeployment",
					},
				},
			},
			expected: &DeploymentDetails{
				Kind:      "SeldonDeployment",
				Namespace: "staging",
				Name:      "sdep-no-spec",
			},
		},
		{
			sdep: &SeldonDeployment{
				SeldonDeployment: sc.SeldonDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "staging",
						Name:      "sdep-with-image",
					},
					TypeMeta: metav1.TypeMeta{
						Kind: "SeldonDeployment",
					},
					Spec: sc.SeldonDeploymentSpec{
						Predictors: []sc.PredictorSpec{
							{
								ComponentSpecs: []*sc.SeldonPodSpec{
									{
										Spec: v1.PodSpec{
											Containers: []v1.Container{
												{
													Image: "seldonio/fixed-model:v0.2",
												},
											},
										},
									},
								},
							},
						},
					},
				},
			},
			expected: &DeploymentDetails{
				Kind:       "SeldonDeployment",
				Namespace:  "staging",
				Name:       "sdep-with-image",
				ModelImage: "seldonio/fixed-model:v0.2",
			},
		},
		{
			sdep: &SeldonDeployment{
				SeldonDeployment: sc.SeldonDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "staging",
						Name:      "sdep-no-component-specs",
					},
					TypeMeta: metav1.TypeMeta{
						Kind: "SeldonDeployment",
					},
					Spec: sc.SeldonDeploymentSpec{
						Predictors: []sc.PredictorSpec{
							{
								Replicas:       4,
								ComponentSpecs: []*sc.SeldonPodSpec{},
							},
						},
					},
				},
			},
			expected: &DeploymentDetails{
				Kind:      "SeldonDeployment",
				Namespace: "staging",
				Name:      "sdep-no-component-specs",
				Replicas:  4,
			},
		},
	}

	for _, test := range tests {
		det := test.sdep.Details()

		g.Expect(det).To(Equal(test.expected))
	}
}

// nolint:funlen
func TestSeldonDeploymentWithDetails(t *testing.T) {
	g := NewGomegaWithT(t)

	tests := []struct {
		sdep     *SeldonDeployment
		det      *DeploymentDetails
		expected *SeldonDeployment
	}{
		{
			sdep: &SeldonDeployment{
				SeldonDeployment: sc.SeldonDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "staging",
						Name:      "sdep-update-namespace",
					},
					TypeMeta: metav1.TypeMeta{
						Kind: "SeldonDeployment",
					},
				},
			},
			det: &DeploymentDetails{
				Namespace: "production",
			},
			expected: &SeldonDeployment{
				SeldonDeployment: sc.SeldonDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "production",
						Name:      "sdep-update-namespace",
					},
					TypeMeta: metav1.TypeMeta{
						Kind: "SeldonDeployment",
					},
				},
			},
		},
		{
			sdep: &SeldonDeployment{
				SeldonDeployment: sc.SeldonDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "staging",
						Name:      "sdep-update-image",
					},
					TypeMeta: metav1.TypeMeta{
						Kind: "SeldonDeployment",
					},
					Spec: sc.SeldonDeploymentSpec{
						Predictors: []sc.PredictorSpec{
							{
								ComponentSpecs: []*sc.SeldonPodSpec{
									{
										Spec: v1.PodSpec{
											Containers: []v1.Container{
												{
													Image: "seldonio/fixed-model:v0.2",
												},
											},
										},
									},
								},
							},
						},
					},
				},
			},
			det: &DeploymentDetails{
				ModelImage: "seldonio/fixed-model:v1.0",
				Replicas:   4,
			},
			expected: &SeldonDeployment{
				SeldonDeployment: sc.SeldonDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "staging",
						Name:      "sdep-update-image",
					},
					TypeMeta: metav1.TypeMeta{
						Kind: "SeldonDeployment",
					},
					Spec: sc.SeldonDeploymentSpec{
						Predictors: []sc.PredictorSpec{
							{
								ComponentSpecs: []*sc.SeldonPodSpec{
									{
										Spec: v1.PodSpec{
											Containers: []v1.Container{
												{
													Image: "seldonio/fixed-model:v1.0",
												},
											},
										},
									},
								},
								Replicas: 4,
							},
						},
					},
				},
			},
		},
		{
			sdep: &SeldonDeployment{
				SeldonDeployment: sc.SeldonDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "staging",
						Name:      "sdep-no-component-specs",
					},
					TypeMeta: metav1.TypeMeta{
						Kind: "SeldonDeployment",
					},
					Spec: sc.SeldonDeploymentSpec{
						Predictors: []sc.PredictorSpec{
							{
								ComponentSpecs: []*sc.SeldonPodSpec{},
							},
						},
					},
				},
			},
			det: &DeploymentDetails{
				ModelImage: "seldonio/should-error:v0.1",
			},
			expected: nil,
		},
	}

	for _, test := range tests {
		dep, err := test.sdep.WithDetails(test.det)

		if test.expected != nil {
			g.Expect(dep).To(Equal(test.expected))
			g.Expect(dep).ToNot(Equal(test.sdep))
			g.Expect(err).ToNot(HaveOccurred())
		} else {
			g.Expect(dep).To(BeNil())
			g.Expect(err).To(HaveOccurred())
		}
	}
}

// nolint:funlen
func TestSeldonDeploymenWithoutStatust(t *testing.T) {
	g := NewGomegaWithT(t)

	tests := []struct {
		sdep     *SeldonDeployment
		expected *SeldonDeployment
	}{
		{
			sdep: &SeldonDeployment{
				SeldonDeployment: sc.SeldonDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "staging",
						Name:      "sdep-with-status",
						SelfLink:  "/api/sdep/sdep-with-status",
						Annotations: map[string]string{
							"kubectl.kubernetes.io/last-applied-configuration": "foo",
						},
					},
					TypeMeta: metav1.TypeMeta{
						Kind: "SeldonDeployment",
					},
					Status: sc.SeldonDeploymentStatus{
						State: "running",
					},
				},
			},
			expected: &SeldonDeployment{
				SeldonDeployment: sc.SeldonDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Namespace:   "staging",
						Name:        "sdep-with-status",
						Annotations: map[string]string{},
					},
					TypeMeta: metav1.TypeMeta{
						Kind: "SeldonDeployment",
					},
				},
			},
		},
	}

	for _, test := range tests {
		dep := test.sdep.WithoutStatus()

		g.Expect(dep).To(Equal(test.expected))
		g.Expect(dep).ToNot(Equal(test.sdep))
	}
}
