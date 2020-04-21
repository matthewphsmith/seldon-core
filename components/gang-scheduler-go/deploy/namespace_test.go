package deploy

import (
	"testing"

	. "github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestNamespaceSkipSSL(t *testing.T) {
	g := NewGomegaWithT(t)

	tests := []struct {
		ns   *Namespace
		skip bool
	}{
		{
			ns: &Namespace{
				Namespace: v1.Namespace{
					ObjectMeta: metav1.ObjectMeta{
						Annotations: map[string]string{
							"skip-ssl": "true",
						},
					},
				},
			},
			skip: true,
		},
		{
			ns:   &Namespace{},
			skip: true,
		},
		{
			ns: &Namespace{
				Namespace: v1.Namespace{
					ObjectMeta: metav1.ObjectMeta{
						Annotations: map[string]string{
							"skip-ssl": "fff",
						},
					},
				},
			},
			skip: true,
		},
		{
			ns: &Namespace{
				Namespace: v1.Namespace{
					ObjectMeta: metav1.ObjectMeta{
						Annotations: map[string]string{
							"skip-ssl": "false",
						},
					},
				},
			},
			skip: false,
		},
	}

	for _, test := range tests {
		skip := test.ns.SkipSSL()

		g.Expect(skip).To(Equal(test.skip))
	}
}
