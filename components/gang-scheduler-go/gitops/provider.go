package gitops

import (
	"github.com/SeldonIO/seldon-deploy-cli/deploy"
)

// GitProvider is an interface to access vendor-specific Git features, like
// PRs, etc.
type GitProvider interface {
	CreatePromotionRequest(*Request) error
	FindPromotionRequest(*Promotion) (*Request, error)
}

// NewGitProvider will instantiate an implementation of a GitOps provider based
// on a given namespace / environment.
func NewGitProvider(creds *Credentials, ns *deploy.Namespace) (GitProvider, error) {
	// TODO: Add support for more GitOps providers.
	return NewBitbucketServerProvider(creds, ns)
}
