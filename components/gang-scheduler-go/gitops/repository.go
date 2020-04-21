package gitops

import (
	"os"

	"github.com/SeldonIO/seldon-deploy-cli/deploy"

	"crypto/tls"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"path"
	"path/filepath"

	"gopkg.in/src-d/go-git.v4"
	"gopkg.in/src-d/go-git.v4/config"
	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/transport/client"
	githttp "gopkg.in/src-d/go-git.v4/plumbing/transport/http"
)

const remoteName = "origin"

// Repository allows us to interact with any GitOps repo to perform promotions
type Repository struct {
	prov  GitProvider
	creds *Credentials
	dir   string
	repo  *git.Repository
	ns    *deploy.Namespace
}

// CloneGitOpsRepository allows you to create a Repository object. During
// creation, it will clone the repo for the given environment using the
// credentials passed as an argument.
func CloneGitOpsRepository(creds *Credentials, ns *deploy.Namespace) (*Repository, error) {
	dir, repo, err := cloneRepo(creds, ns)
	if err != nil {
		return nil, err
	}

	prov, err := NewGitProvider(creds, ns)
	if err != nil {
		return nil, err
	}

	return &Repository{
		prov:  prov,
		creds: creds,
		dir:   dir,
		repo:  repo,
		ns:    ns,
	}, nil
}

func cloneRepo(creds *Credentials, ns *deploy.Namespace) (string, *git.Repository, error) {
	url := ns.RepositoryURL()

	basePath := fmt.Sprintf("%s-", path.Base(url))

	dir, err := ioutil.TempDir("", basePath)
	if err != nil {
		return "", nil, err
	}

	// TODO: Is there a flag to know if we should skip SSL certificate
	// verification?
	skipClient := &http.Client{
		Transport: &http.Transport{
			// nolint: g402
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		},
	}
	client.InstallProtocol("https", githttp.NewClient(skipClient))

	// Clone the repository to the given directory
	r, err := git.PlainClone(dir, false, &git.CloneOptions{
		Auth:       creds.Auth(),
		URL:        url,
		RemoteName: remoteName,
	})

	return dir, r, err
}

func (r *Repository) Close() error {
	return os.RemoveAll(r.dir)
}

func (r *Repository) RequestPromotion(prom *Promotion) error {
	req, err := r.findOrCreateRequest(prom)
	if err != nil {
		return err
	}

	err = r.checkoutRequest(req)
	if err != nil {
		return err
	}

	err = r.copyDeployment(prom)
	if err != nil {
		return err
	}

	err = r.commitPromotion(prom)
	if err != nil {
		return err
	}

	return r.pushRequest(req)
}

func (r *Repository) findOrCreateRequest(prom *Promotion) (*Request, error) {
	req, err := r.prov.FindPromotionRequest(prom)
	if err != nil {
		return nil, err
	}

	if req == nil {
		req = &Request{
			Promotion: prom,
			Exists:    false,
		}
	}

	return req, nil
}

func (r *Repository) checkoutRequest(req *Request) error {
	wt, err := r.repo.Worktree()
	if err != nil {
		return err
	}

	if req.Exists {
		// If PR exists, we need to fetch the branch.
		// TODO: Fetch only the one we are interested in.
		fo := &git.FetchOptions{
			RefSpecs: []config.RefSpec{"refs/*:refs/*", "HEAD:refs/heads/HEAD"},
			Auth:     r.creds.Auth(),
		}

		err = r.repo.Fetch(fo)
		if err != nil {
			return err
		}
	}

	co, err := r.checkoutOptions(req)
	if err != nil {
		return err
	}

	return wt.Checkout(co)
}

func (r *Repository) checkoutOptions(req *Request) (*git.CheckoutOptions, error) {
	co := &git.CheckoutOptions{
		Branch: plumbing.NewBranchReferenceName(req.Branch()),
	}

	if req.Exists {
		return co, nil
	}

	h, err := r.repo.Head()
	if err != nil {
		return nil, err
	}

	co.Hash = h.Hash()
	co.Create = true

	return co, nil
}

func (r *Repository) copyDeployment(prom *Promotion) error {
	deploymentPath := prom.DeploymentPath()
	absPath := filepath.Join(r.dir, deploymentPath)
	absPathDir := filepath.Dir(absPath)

	err := os.MkdirAll(absPathDir, os.ModePerm)
	if err != nil {
		return err
	}

	enc, err := json.MarshalIndent(prom.Deployment, "", "    ")
	if err != nil {
		return err
	}

	return ioutil.WriteFile(absPath, enc, 0644)
}

func (r *Repository) commitPromotion(prom *Promotion) error {
	deploymentPath := prom.DeploymentPath()

	wt, err := r.repo.Worktree()
	if err != nil {
		return err
	}

	_, err = wt.Add(deploymentPath)
	if err != nil {
		return err
	}

	commit, err := wt.Commit(
		prom.CommitMessage(),
		&git.CommitOptions{Author: prom.CommitAuthor()},
	)
	if err != nil {
		return err
	}

	_, err = r.repo.CommitObject(commit)

	return err
}

func (r *Repository) pushRequest(req *Request) error {
	// First we need to push changes
	err := r.repo.Push(&git.PushOptions{Auth: r.creds.Auth()})
	if err != nil {
		return err
	}

	if req.Exists {
		// If request already exists and is opened, no need to do anything else
		return nil
	}

	return r.prov.CreatePromotionRequest(req)
}
