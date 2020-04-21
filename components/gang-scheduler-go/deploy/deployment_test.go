package deploy

import (
	"encoding/json"
	"io/ioutil"
	"testing"

	. "github.com/onsi/gomega"
)

// nolint:funlen
func TestDeploymentUnmarshall(t *testing.T) {
	g := NewGomegaWithT(t)

	tests := []struct {
		fixture  string
		expected *DeploymentDetails
	}{
		{
			fixture: "../testdata/SeldonDeployment.json",
			expected: &DeploymentDetails{
				Kind: "SeldonDeployment",
				Name: "seldon-deployment-example",
			},
		},
	}

	for _, test := range tests {
		file, err := ioutil.ReadFile(test.fixture)
		g.Expect(err).ToNot(HaveOccurred())

		var sdep SeldonDeployment
		err = json.Unmarshal(file, &sdep)

		det := sdep.Details()

		g.Expect(err).ToNot(HaveOccurred())
		g.Expect(det.Name).To(Equal(test.expected.Name))
		g.Expect(det.Kind).To(Equal(test.expected.Kind))
	}
}

func TestNewDeployment(t *testing.T) {
	g := NewGomegaWithT(t)

	dep, err := NewDeployment("../testdata/SeldonDeployment.json")
	g.Expect(err).ToNot(HaveOccurred())
	g.Expect(dep).To(BeAssignableToTypeOf(&SeldonDeployment{}))

	det := dep.Details()

	g.Expect(det.Name).To(Equal("seldon-deployment-example"))
	g.Expect(det.Kind).To(Equal("SeldonDeployment"))
}
