package gitops

import (
	"testing"

	. "github.com/onsi/gomega"
	"gopkg.in/src-d/go-git.v4/plumbing/transport"
	"gopkg.in/src-d/go-git.v4/plumbing/transport/http"
)

func TestCredentialsAuth(t *testing.T) {
	g := NewGomegaWithT(t)

	tests := []struct {
		creds    *Credentials
		expected transport.AuthMethod
	}{
		{
			creds: &Credentials{
				User:  "test-user",
				Token: "12341234",
			},
			expected: &http.BasicAuth{
				Username: "test-user",
				Password: "12341234",
			},
		},
	}

	for _, test := range tests {
		auth := test.creds.Auth()

		g.Expect(auth).To(Equal(test.expected))
	}
}
