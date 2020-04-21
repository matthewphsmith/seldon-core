package gitops

import (
	"testing"

	"github.com/SeldonIO/seldon-deploy-cli/deploy"
	. "github.com/onsi/gomega"
	sc "github.com/seldonio/seldon-core/operator/apis/machinelearning/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestPromotionCommitMessage(t *testing.T) {
	g := NewGomegaWithT(t)

	tests := []struct {
		prom     *Promotion
		expected string
	}{
		{
			prom: &Promotion{
				To: &deploy.Namespace{
					Namespace: v1.Namespace{
						ObjectMeta: metav1.ObjectMeta{Name: "production"},
					},
				},
				Author: "Data Scientist 1",
				Email:  "ds1@org.com",
			},
			expected: `{
        "Action":"Moving deployment to production",
        "Message":"",
        "Author":"Data Scientist 1",
        "Email":"ds1@org.com"}`,
		},
	}

	for _, test := range tests {
		p := test.prom.CommitMessage()

		g.Expect(p).To(MatchJSON(test.expected))
	}
}

func TestPromotionDeploymentPath(t *testing.T) {
	g := NewGomegaWithT(t)

	tests := []struct {
		prom     *Promotion
		expected string
	}{
		{
			prom: &Promotion{
				To: &deploy.Namespace{
					Namespace: v1.Namespace{
						ObjectMeta: metav1.ObjectMeta{Name: "production"},
					},
				},
				Deployment: &deploy.SeldonDeployment{
					SeldonDeployment: sc.SeldonDeployment{
						ObjectMeta: metav1.ObjectMeta{Name: "sdep-1"},
						TypeMeta:   metav1.TypeMeta{Kind: "SeldonDeployment"},
					},
				},
			},
			expected: "production/SeldonDeployment/sdep-1/sdep-1.json",
		},
	}

	for _, test := range tests {
		p := test.prom.DeploymentPath()

		g.Expect(p).To(Equal(test.expected))
	}
}
