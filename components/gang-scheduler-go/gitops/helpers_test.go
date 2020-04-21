package gitops

import (
	"testing"

	. "github.com/onsi/gomega"
)

func TestRepositoryDetails(t *testing.T) {
	g := NewGomegaWithT(t)

	tests := []struct {
		repoURL string
		owner   string
		slug    string
		err     error
	}{
		{
			repoURL: "https://bitbucket.com/scm/seldonio/seldon-gitops.git",
			owner:   "seldonio",
			slug:    "seldon-gitops",
		},
		{
			repoURL: "https://bitbucket.com/scm/seldonio/seldon-gitops",
			owner:   "seldonio",
			slug:    "seldon-gitops",
		},
		{
			repoURL: "https://bitbucket.com/scm/seldonio/seldon.gitops",
			owner:   "seldonio",
			slug:    "seldon.gitops",
		},
		{
			repoURL: "https://bitbucket.com/scm/seldonio/seldon.gitops.git",
			owner:   "seldonio",
			slug:    "seldon.gitops",
		},
		{
			repoURL: "https://seldondev@bitbucket.org/seldonio/seldon-gitops.git",
			owner:   "seldonio",
			slug:    "seldon-gitops",
		},
	}

	for _, test := range tests {
		owner, slug, err := RepositoryDetails(test.repoURL)

		if test.err != nil {
			g.Expect(err).To(HaveOccurred())
		} else {
			g.Expect(err).ToNot(HaveOccurred())
			g.Expect(owner).To(Equal(test.owner))
			g.Expect(slug).To(Equal(test.slug))
		}
	}
}
