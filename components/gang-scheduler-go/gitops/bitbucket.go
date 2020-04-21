package gitops

import (
	"github.com/ktrysmt/go-bitbucket"
)

type BitbucketProvider struct {
	GitProvider

	c *bitbucket.Client
}

func NewBitbucketProvider(creds *Credentials) *BitbucketProvider {
	return &BitbucketProvider{
		c: bitbucket.NewBasicAuth(creds.User, creds.Token),
	}
}

func (b *BitbucketProvider) CreatePromotionRequest(req *Request) error {
	repoURL := req.Promotion.To.RepositoryURL()

	owner, slug, err := RepositoryDetails(repoURL)
	if err != nil {
		return err
	}

	po := &bitbucket.PullRequestsOptions{
		Owner:             owner,
		RepoSlug:          slug,
		SourceBranch:      req.Branch(),
		DestinationBranch: "master",
		Title:             req.Title(),
		CloseSourceBranch: true,
	}

	_, err = b.c.Repositories.PullRequests.Create(po)

	return err
}
