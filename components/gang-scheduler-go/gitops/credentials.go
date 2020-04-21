package gitops

import (
	"gopkg.in/src-d/go-git.v4/plumbing/transport"
	"gopkg.in/src-d/go-git.v4/plumbing/transport/http"
)

// Credentials is a holder for cred info to access the GitOps repo
type Credentials struct {
	User  string
	Token string
}

func (c *Credentials) Auth() transport.AuthMethod {
	return &http.BasicAuth{
		Username: c.User,
		Password: c.Token,
	}
}
