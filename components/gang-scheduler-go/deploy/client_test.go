package deploy

import (
	"net/http"
	"testing"

	. "github.com/onsi/gomega"
	sc "github.com/seldonio/seldon-core/operator/apis/machinelearning/v1"
	"gopkg.in/h2non/gock.v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	testNamespace = "staging"
	testServerURL = "http://fake.com/seldon-deploy"
)

func clientFixture() *Client {
	c, _ := NewClient(testServerURL, authFixture())
	return c
}

func TestGetNamespaces(t *testing.T) {
	g := NewGomegaWithT(t)

	defer gock.Off()

	expectedNs := []*Namespace{
		{
			Namespace: v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{Name: "staging"},
			},
		},
		{
			Namespace: v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{Name: "production"},
			},
		},
	}
	cluster := &Cluster{Namespaces: expectedNs}

	// We don't use gomega/ghttp because it's not yet compatible with
	// NewGomegaWithT: https://github.com/onsi/gomega/issues/321
	gock.New(testServerURL).
		Get("/api/cluster").
		Reply(http.StatusOK).
		JSON(cluster)

	client := clientFixture()
	ns, err := client.GetNamespaces()

	g.Expect(err).ToNot(HaveOccurred())
	g.Expect(ns).To(Equal(expectedNs))
}

func TestGetNamespace(t *testing.T) {
	g := NewGomegaWithT(t)

	defer gock.Off()

	needle := &Namespace{
		Namespace: v1.Namespace{
			ObjectMeta: metav1.ObjectMeta{Name: "production"},
		},
	}

	namespaces := []*Namespace{
		{
			Namespace: v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{Name: "staging"},
			},
		},
		needle,
	}
	cluster := &Cluster{Namespaces: namespaces}

	// We don't use gomega/ghttp because it's not yet compatible with
	// NewGomegaWithT: https://github.com/onsi/gomega/issues/321
	gock.New(testServerURL).
		Get("/api/cluster").
		Reply(http.StatusOK).
		JSON(cluster)

	client := clientFixture()
	ns, err := client.GetNamespace("production")

	g.Expect(err).ToNot(HaveOccurred())
	g.Expect(ns).To(Equal(needle))
}

func TestGetDeployments(t *testing.T) {
	g := NewGomegaWithT(t)

	defer gock.Off()

	expectedDeps := []Deployment{
		&SeldonDeployment{
			SeldonDeployment: sc.SeldonDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "sdep-1"},
				TypeMeta:   metav1.TypeMeta{Kind: "SeldonDeployment"},
			},
		},
		&SeldonDeployment{
			SeldonDeployment: sc.SeldonDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "sdep-2"},
				TypeMeta:   metav1.TypeMeta{Kind: "SeldonDeployment"},
			},
		},
	}

	// We don't use gomega/ghttp because it's not yet compatible with
	// NewGomegaWithT: https://github.com/onsi/gomega/issues/321
	gock.New(testServerURL).
		Get("/api/deployments").
		MatchParam("namespace", testNamespace).
		Reply(http.StatusOK).
		JSON(expectedDeps)

	client := clientFixture()
	deps, err := client.GetDeployments(testNamespace)

	g.Expect(err).ToNot(HaveOccurred())
	g.Expect(deps).To(Equal(expectedDeps))
}

func TestGetDeployment(t *testing.T) {
	g := NewGomegaWithT(t)

	defer gock.Off()

	needle := &SeldonDeployment{
		SeldonDeployment: sc.SeldonDeployment{
			ObjectMeta: metav1.ObjectMeta{Name: "sdep-needle"},
			TypeMeta:   metav1.TypeMeta{Kind: "SeldonDeployment"},
		},
	}

	deps := []Deployment{
		&SeldonDeployment{
			SeldonDeployment: sc.SeldonDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "sdep-1"},
				TypeMeta:   metav1.TypeMeta{Kind: "SeldonDeployment"},
			},
		},
		&SeldonDeployment{
			SeldonDeployment: sc.SeldonDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "sdep-2"},
				TypeMeta:   metav1.TypeMeta{Kind: "SeldonDeployment"},
			},
		},
		needle,
	}

	// We don't use gomega/ghttp because it's not yet compatible with
	// NewGomegaWithT: https://github.com/onsi/gomega/issues/321
	gock.New(testServerURL).
		Get("/api/deployments").
		MatchParam("namespace", testNamespace).
		Reply(http.StatusOK).
		JSON(deps)

	client := clientFixture()
	dep, err := client.GetDeployment(testNamespace, "sdep-needle")

	g.Expect(err).ToNot(HaveOccurred())
	g.Expect(dep).To(Equal(needle))
}

func TestUpdateDeployment(t *testing.T) {
	g := NewGomegaWithT(t)

	defer gock.Off()

	dep := &SeldonDeployment{
		SeldonDeployment: sc.SeldonDeployment{
			ObjectMeta: metav1.ObjectMeta{Name: "iris-classifier"},
			TypeMeta:   metav1.TypeMeta{Kind: "SeldonDeployment"},
			Spec: sc.SeldonDeploymentSpec{
				Name: "iris-classifier",
				Predictors: []sc.PredictorSpec{
					{
						Name:     "default",
						Replicas: 2, // nolint: mnd
					},
				},
			},
		},
	}

	// We don't use gomega/ghttp because it's not yet compatible with
	// NewGomegaWithT: https://github.com/onsi/gomega/issues/321
	serverURL := "http://fake.com/seldon-deploy"
	gock.New(serverURL).
		Put("/api/deployments").
		MatchParam("type", dep.Kind).
		MatchType("form").
		Reply(http.StatusOK)

	client := clientFixture()
	err := client.UpdateDeployment(dep)

	g.Expect(err).ToNot(HaveOccurred())
}
