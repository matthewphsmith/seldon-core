package gitops

import (
	"fmt"
	"testing"

	"github.com/SeldonIO/seldon-deploy-cli/deploy"
	"github.com/google/uuid"
	. "github.com/onsi/gomega"
	sc "github.com/seldonio/seldon-core/operator/apis/machinelearning/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// nolint:funlen
func TestRequestMetadataIsEqual(t *testing.T) {
	g := NewGomegaWithT(t)

	u1 := uuid.New()
	u2 := uuid.New()
	tests := []struct {
		a     *RequestMetadata
		b     *RequestMetadata
		equal bool
	}{
		{
			a: &RequestMetadata{
				Action:     "something something",
				To:         "production",
				Deployment: "my-model",
			},
			b: &RequestMetadata{
				Action:     "something else",
				To:         "production",
				Deployment: "my-model",
			},
			equal: true,
		},
		{
			a: &RequestMetadata{
				ID:         &u1,
				To:         "diff but same id",
				Deployment: "diff but same id",
			},
			b: &RequestMetadata{
				ID:         &u1,
				To:         "production",
				Deployment: "my-model",
			},
			equal: true,
		},
		{
			a: &RequestMetadata{
				ID:         &u1,
				To:         "production",
				Deployment: "my-model",
			},
			b: &RequestMetadata{
				ID:         &u2,
				To:         "production",
				Deployment: "my-model",
			},
			equal: false,
		},
		{
			a: &RequestMetadata{
				To:         "production",
				Deployment: "other-model",
			},
			b: &RequestMetadata{
				To:         "production",
				Deployment: "my-model",
			},
			equal: false,
		},
		{
			a: &RequestMetadata{
				To:         "other-env",
				Deployment: "my-model",
			},
			b: &RequestMetadata{
				To:         "production",
				Deployment: "my-model",
			},
			equal: false,
		},
	}

	for _, test := range tests {
		equal := test.a.IsEqual(test.b)

		g.Expect(equal).To(Equal(test.equal))
	}
}

func TestRequestBranch(t *testing.T) {
	g := NewGomegaWithT(t)

	u1 := uuid.New()
	reqs := []*Request{
		{ID: &u1},
		{},
	}

	for _, req := range reqs {
		b := req.Branch()
		expected := fmt.Sprintf("promotion/%s", req.ID)

		g.Expect(b).To(Equal(expected))
	}
}

// nolint:funlen
func TestRequestMetadata(t *testing.T) {
	g := NewGomegaWithT(t)

	u1 := uuid.New()
	tests := []struct {
		req  *Request
		meta *RequestMetadata
	}{
		{
			req: &Request{
				Promotion: &Promotion{
					To: &deploy.Namespace{
						Namespace: v1.Namespace{
							ObjectMeta: metav1.ObjectMeta{
								Name: "production",
							},
						},
					},
					Deployment: &deploy.SeldonDeployment{
						SeldonDeployment: sc.SeldonDeployment{
							ObjectMeta: metav1.ObjectMeta{
								Name: "my-model",
							},
						},
					},
				},
			},
			meta: &RequestMetadata{
				Action:     "Promotion to production",
				To:         "production",
				Deployment: "my-model",
			},
		},
		{
			req: &Request{
				ID: &u1,
				Promotion: &Promotion{
					To: &deploy.Namespace{
						Namespace: v1.Namespace{
							ObjectMeta: metav1.ObjectMeta{
								Name: "production",
							},
						},
					},
					Deployment: &deploy.SeldonDeployment{
						SeldonDeployment: sc.SeldonDeployment{
							ObjectMeta: metav1.ObjectMeta{
								Name: "my-model",
							},
						},
					},
				},
			},
			meta: &RequestMetadata{
				ID:         &u1,
				Action:     "Promotion to production",
				To:         "production",
				Deployment: "my-model",
			},
		},
	}

	for _, test := range tests {
		meta := test.req.Metadata()
		g.Expect(meta).To(Equal(test.meta))
	}
}

func TestRequestDescription(t *testing.T) {
	g := NewGomegaWithT(t)

	tests := []struct {
		req  *Request
		desc string
	}{
		{
			req: &Request{
				Promotion: &Promotion{
					To: &deploy.Namespace{
						Namespace: v1.Namespace{
							ObjectMeta: metav1.ObjectMeta{
								Name: "production",
							},
						},
					},
					Deployment: &deploy.SeldonDeployment{
						SeldonDeployment: sc.SeldonDeployment{
							ObjectMeta: metav1.ObjectMeta{
								Name: "my-model",
							},
						},
					},
				},
			},
			desc: `{
        "Action": "Promotion to production",
        "To": "production",
        "Deployment": "my-model"}`,
		},
	}

	for _, test := range tests {
		desc := test.req.Description()
		g.Expect(desc).To(MatchJSON(test.desc))
	}
}
