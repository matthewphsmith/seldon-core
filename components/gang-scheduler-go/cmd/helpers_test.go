package cmd

import (
	"testing"

	. "github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
)

func TestToEnvList(t *testing.T) {
	g := NewGomegaWithT(t)

	tests := []struct {
		flags   []string
		envList []v1.EnvVar
	}{
		{
			flags: []string{"foo=bar", "foo2=bar2"},
			envList: []v1.EnvVar{
				{
					Name:  "foo",
					Value: "bar",
				},
				{
					Name:  "foo2",
					Value: "bar2",
				},
			},
		},
		{
			flags:   []string{},
			envList: []v1.EnvVar{},
		},
		{
			flags:   []string{"foo=bar", "invalid=key=value"},
			envList: nil,
		},
		{
			flags:   []string{"foo=bar", "invalid"},
			envList: nil,
		},
	}

	for _, test := range tests {
		envList, err := toEnvList(test.flags)
		if test.envList != nil {
			g.Expect(err).ToNot(HaveOccurred())
			g.Expect(envList).To(Equal(test.envList))
		} else {
			g.Expect(err).To(HaveOccurred())
		}
	}
}
