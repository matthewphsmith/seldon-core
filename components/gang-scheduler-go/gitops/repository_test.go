package gitops

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/src-d/go-git.v4"
	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/object"

	"github.com/SeldonIO/seldon-deploy-cli/deploy"
	"github.com/google/uuid"
	. "github.com/onsi/gomega"
	sc "github.com/seldonio/seldon-core/operator/apis/machinelearning/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func repoFixture() *Repository {
	dir, _ := ioutil.TempDir("", "test-")
	repo, _ := git.PlainInit(dir, false)

	// Add README.md
	fname := "README.md"
	fpath := filepath.Join(dir, fname)
	os.Create(fpath) // nolint: errcheck

	wt, _ := repo.Worktree()
	wt.Add(fname) // nolint: errcheck

	commit, _ := wt.Commit(
		"Initial Commit",
		&git.CommitOptions{
			Author: &object.Signature{
				Name:  "Test",
				Email: "test@seldon.io",
			},
		},
	)

	repo.CommitObject(commit) // nolint: errcheck

	return &Repository{dir: dir, repo: repo}
}

func TestRepositoryCheckoutRequest(t *testing.T) {
	g := NewGomegaWithT(t)

	r := repoFixture()
	defer r.Close()

	req := &Request{}

	err := r.checkoutRequest(req)

	g.Expect(err).ToNot(HaveOccurred())

	// Check current branch is promotion
	branch := ""

	h, _ := r.repo.Head()
	bs, _ := r.repo.Branches()
	// nolint:errcheck
	bs.ForEach(func(b *plumbing.Reference) error {
		if b.Hash() == h.Hash() {
			branch = b.Name().Short()
		}

		return nil
	})

	g.Expect(branch).To(Equal(req.Branch()))
}

func TestRepositoryCheckoutOptions(t *testing.T) {
	g := NewGomegaWithT(t)

	repo := repoFixture()
	u1 := uuid.New()
	req := &Request{ID: &u1}

	tests := []struct {
		exists bool
		opts   *git.CheckoutOptions
	}{
		{
			exists: true,
			opts: &git.CheckoutOptions{
				Branch: plumbing.NewBranchReferenceName(req.Branch()),
			},
		},
		{
			exists: false,
			opts: &git.CheckoutOptions{
				Branch: plumbing.NewBranchReferenceName(req.Branch()),
				Create: true,
			},
		},
	}

	for _, test := range tests {
		req.Exists = test.exists
		opts, err := repo.checkoutOptions(req)

		g.Expect(err).ToNot(HaveOccurred())
		g.Expect(opts.Create).To(Equal(test.opts.Create))
		g.Expect(opts.Branch).To(Equal(test.opts.Branch))
	}
}

func TestRepositoryCopyDeployment(t *testing.T) {
	g := NewGomegaWithT(t)

	r := repoFixture()
	defer r.Close()

	prom := &Promotion{
		Deployment: &deploy.SeldonDeployment{
			SeldonDeployment: sc.SeldonDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "sdep-1"},
				TypeMeta:   metav1.TypeMeta{Kind: "SeldonDeployment"},
			},
		},
		To: &deploy.Namespace{
			Namespace: v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{Name: "production"},
			},
		},
	}
	err := r.copyDeployment(prom)

	g.Expect(err).ToNot(HaveOccurred())
	g.Expect(r.dir).To(BeADirectory())

	absPath := filepath.Join(r.dir, prom.DeploymentPath())
	g.Expect(absPath).To(BeARegularFile())
}

func TestRepositoryCommitPromotion(t *testing.T) {
	g := NewGomegaWithT(t)

	r := repoFixture()
	defer r.Close()

	prom := &Promotion{
		Deployment: &deploy.SeldonDeployment{
			SeldonDeployment: sc.SeldonDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "sdep-1"},
				TypeMeta:   metav1.TypeMeta{Kind: "SeldonDeployment"},
			},
		},
		Author: "Data Scientist",
		Email:  "ds1@org.com",
		To: &deploy.Namespace{
			Namespace: v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{Name: "production"},
			},
		},
	}

	r.copyDeployment(prom) // nolint: errcheck

	err := r.commitPromotion(prom)

	g.Expect(err).ToNot(HaveOccurred())

	// Get last commit
	commits, _ := r.repo.CommitObjects()
	defer commits.Close()

	commit, err := commits.Next()

	g.Expect(err).ToNot(HaveOccurred())
	g.Expect(commit.Message).To(MatchJSON(prom.CommitMessage()))
	g.Expect(commit.Author.Name).To(Equal(prom.Author))
	g.Expect(commit.Author.Email).To(Equal(prom.Email))

	// Assert file is there
	deploymentPath := prom.DeploymentPath()
	file, err := commit.File(deploymentPath)

	g.Expect(err).ToNot(HaveOccurred())
	g.Expect(file.Name).To(Equal(deploymentPath))
}

func TestRepositoryClean(t *testing.T) {
	g := NewGomegaWithT(t)

	r := repoFixture()

	g.Expect(r.dir).To(BeADirectory())

	err := r.Close()

	g.Expect(err).ToNot(HaveOccurred())
	g.Expect(r.dir).ToNot(BeADirectory())
}
