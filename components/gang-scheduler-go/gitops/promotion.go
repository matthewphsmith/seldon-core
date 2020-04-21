package gitops

import (
	"fmt"
	"path/filepath"
	"time"

	"github.com/SeldonIO/seldon-deploy-cli/deploy"
	"gopkg.in/src-d/go-git.v4/plumbing/object"
)

// Promotion between environments of a deployment
type Promotion struct {
	Deployment deploy.Deployment
	From       *deploy.Namespace
	To         *deploy.Namespace
	Author     string
	Email      string
}

func (p *Promotion) CommitMessage() string {
	return fmt.Sprintf(
		`{"Action":"Moving deployment to %s","Message":"","Author":"%s","Email":"%s"}`,
		p.To.Name,
		p.Author,
		p.Email,
	)
}

func (p *Promotion) CommitAuthor() *object.Signature {
	return &object.Signature{
		Name:  p.Author,
		Email: p.Email,
		When:  time.Now(),
	}
}

func (p *Promotion) DeploymentPath() string {
	det := p.Deployment.Details()

	env := p.To.Name
	kind := det.Kind
	name := det.Name

	filename := fmt.Sprintf("%s.json", name)
	deploymentPath := filepath.Join(env, kind, name, filename)

	return deploymentPath
}
