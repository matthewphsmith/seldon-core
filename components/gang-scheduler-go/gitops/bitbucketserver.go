package gitops

import (
	"context"
	"crypto/tls"
	"fmt"
	"net/http"
	"net/url"

	"github.com/SeldonIO/seldon-deploy-cli/deploy"
	"github.com/SeldonIO/seldon-deploy-cli/log"
	bitbucketv1 "github.com/gfleury/go-bitbucket-v1"
)

type BitbucketServerProvider struct {
	GitProvider

	c *bitbucketv1.APIClient
}

func NewBitbucketServerProvider(creds *Credentials, ns *deploy.Namespace) (*BitbucketServerProvider, error) {
	ctx := context.Background()
	auth := bitbucketv1.BasicAuth{UserName: creds.User, Password: creds.Token}
	authContext := context.WithValue(ctx, bitbucketv1.ContextBasicAuth, auth)

	serverURL, err := getBitbucketServerURL(ns)
	if err != nil {
		return nil, err
	}

	cfg := bitbucketv1.NewConfiguration(serverURL)

	if ns.SkipSSL() {
		cfg.HTTPClient = &http.Client{
			Transport: &http.Transport{
				// nolint: g402
				TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
			},
		}
	}

	prov := &BitbucketServerProvider{
		c: bitbucketv1.NewAPIClient(authContext, cfg),
	}

	return prov, nil
}

func getBitbucketServerURL(ns *deploy.Namespace) (string, error) {
	repoURL := ns.RepositoryURL()

	u, err := url.Parse(repoURL)
	if err != nil {
		return "", err
	}

	serverURL := fmt.Sprintf("%s://%s/rest", u.Scheme, u.Hostname())

	return serverURL, nil
}

func (b *BitbucketServerProvider) CreatePromotionRequest(req *Request) error {
	repoURL := req.Promotion.To.RepositoryURL()

	proj, repo, err := RepositoryDetails(repoURL)
	if err != nil {
		return err
	}

	pr := NewBitbucketServerRequest(req)

	prMap, err := pr.MarshalMap()
	if err != nil {
		return err
	}

	_, err = b.c.DefaultApi.CreatePullRequestWithOptions(proj, repo, prMap)

	return err
}

func (b *BitbucketServerProvider) FindPromotionRequest(prom *Promotion) (*Request, error) {
	repoURL := prom.To.RepositoryURL()

	proj, repo, err := RepositoryDetails(repoURL)
	if err != nil {
		return nil, err
	}

	// we can't filter so we need to iterate through all pages
	page := &BitbucketServerRequestsPage{}
	for !page.IsLastPage {
		o := b.getFindOptions(page.NextPageStart)

		res, err := b.c.DefaultApi.GetPullRequestsPage(proj, repo, o)
		if err != nil {
			return nil, err
		}

		err = page.UnmarshalMap(res.Values)
		if err != nil {
			return nil, err
		}

		prMeta := b.findRequestInPage(prom, page)
		if prMeta != nil {
			// Found! Set ID of request and return.
			req := &Request{
				ID:        prMeta.ID,
				Promotion: prom,
				Exists:    true,
			}

			return req, nil
		}
	}

	return nil, nil
}

func (b *BitbucketServerProvider) findRequestInPage(
	prom *Promotion,
	page *BitbucketServerRequestsPage,
) *RequestMetadata {
	meta := NewRequestMetadata(prom)

	for _, pr := range page.Values {
		prMeta, err := pr.DescriptionMetadata()
		if err != nil {
			// log but ignore and continue
			log.Warningf("Invalid metadata in PR #%d: %s", *pr.ID, err)
			continue
		}

		if meta.IsEqual(prMeta) {
			return prMeta
		}
	}

	return nil
}

func (b *BitbucketServerProvider) getFindOptions(start int) map[string]interface{} {
	return map[string]interface{}{
		"state":          "OPEN",
		"order":          "NEWEST",
		"start":          start,
		"withAttributes": false,
		"withProperties": false,
	}
}
