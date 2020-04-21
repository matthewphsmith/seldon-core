package gitops

import (
	"fmt"
	"net/http"
	"strconv"
	"testing"

	"github.com/SeldonIO/seldon-deploy-cli/deploy"
	"github.com/google/uuid"
	. "github.com/onsi/gomega"
	sc "github.com/seldonio/seldon-core/operator/apis/machinelearning/v1"
	"gopkg.in/h2non/gock.v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const testBitbucketServerURL = "http://bitbucket.seldon.io"

// nolint:gochecknoglobals
var testBitbucketServerCredentials = &Credentials{
	User:  "bitbucket-server-admin",
	Token: "12341234",
}

func TestGetBitbucketServerURL(t *testing.T) {
	g := NewGomegaWithT(t)

	tests := []struct {
		repoURL   string
		serverURL string
	}{
		{
			repoURL:   "https://bitbucket.seldon.io/scm/deploy/sd.git",
			serverURL: "https://bitbucket.seldon.io/rest",
		},
		{
			repoURL:   "https://bitbucket.seldon.io:7999/scm/deploy/sd.git",
			serverURL: "https://bitbucket.seldon.io/rest",
		},
	}

	for _, test := range tests {
		ns := &deploy.Namespace{
			Namespace: v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"git-repo": test.repoURL,
					},
				},
			},
		}

		serverURL, err := getBitbucketServerURL(ns)

		g.Expect(err).ToNot(HaveOccurred())
		g.Expect(serverURL).To(Equal(test.serverURL))
	}
}

func TestBitbucketServerCreatePromotionRequest(t *testing.T) {
	g := NewGomegaWithT(t)

	defer gock.Off()

	req := &Request{
		Promotion: &Promotion{
			To: &deploy.Namespace{
				Namespace: v1.Namespace{
					ObjectMeta: metav1.ObjectMeta{
						Name: "production",
						Annotations: map[string]string{
							"git-repo": fmt.Sprintf(
								"%s/scm/seldonio/seldon-gitops.git",
								testBitbucketServerURL,
							),
							"skip-ssl": "false",
						},
					},
				},
			},
			Deployment: &deploy.SeldonDeployment{
				SeldonDeployment: sc.SeldonDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name: "my-model",
					},
				},
			},
		},
	}

	pr := NewBitbucketServerRequest(req)

	// We don't use gomega/ghttp because it's not yet compatible with
	// NewGomegaWithT: https://github.com/onsi/gomega/issues/321
	gock.New(testBitbucketServerURL).
		Post("/rest/api/1.0/projects/seldonio/repos/seldon-gitops/pull-requests").
		BasicAuth(testBitbucketServerCredentials.User, testBitbucketServerCredentials.Token).
		Reply(http.StatusCreated).
		JSON(pr)

	b, _ := NewBitbucketServerProvider(testBitbucketServerCredentials, req.Promotion.To)
	err := b.CreatePromotionRequest(req)

	g.Expect(err).ToNot(HaveOccurred())
}

// nolint:funlen
func TestBitbucketServerFindPromotionRequest(t *testing.T) {
	g := NewGomegaWithT(t)

	defer gock.Off()

	prom := &Promotion{
		To: &deploy.Namespace{
			Namespace: v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "production",
					Annotations: map[string]string{
						"git-repo": fmt.Sprintf(
							"%s/scm/seldonio/seldon-gitops.git",
							testBitbucketServerURL,
						),
						"skip-ssl": "false",
					},
				},
			},
		},
		Deployment: &deploy.SeldonDeployment{
			SeldonDeployment: sc.SeldonDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name: "my-model",
				},
			},
		},
	}

	id2 := 2
	id3 := 3

	prID := 4
	reqID := uuid.New()
	pr := NewBitbucketServerRequest(&Request{
		ID:        &reqID,
		Promotion: prom,
	})
	pr.ID = &prID

	tests := []struct {
		pages []*BitbucketServerRequestsPage
		found bool
	}{
		{
			pages: []*BitbucketServerRequestsPage{
				{
					Values: []*BitbucketServerRequest{
						{
							ID:          &id2,
							Description: "faulty meta",
						},
						{
							ID:          &id3,
							Description: `{"ID":"c01d7cf6-ec3f-47f0-9556-a5d6e9009a43","To":"staging","Deployment":"something-else"}`,
						},
						pr,
					},
				},
			},
			found: true,
		},
		{
			pages: []*BitbucketServerRequestsPage{
				{
					Values: []*BitbucketServerRequest{
						{
							ID:          &id3,
							Description: `{"ID":"c01d7cf6-ec3f-47f0-9556-a5d6e9009a43","To":"staging","Deployment":"something-else"}`,
						},
					},
					NextPageStart: 1,
				},
				{
					Values: []*BitbucketServerRequest{
						pr,
					},
				},
			},
			found: true,
		},
		{
			pages: []*BitbucketServerRequestsPage{
				{
					Values: []*BitbucketServerRequest{
						{
							ID:          &id3,
							Description: `{"ID":"c01d7cf6-ec3f-47f0-9556-a5d6e9009a43","To":"staging","Deployment":"something-else"}`,
						},
					},
					NextPageStart: 1,
				},
				{
					Values: []*BitbucketServerRequest{
						{
							ID:          &id2,
							Description: `{"ID":"c01d7cf6-ec3f-47f0-9556-a5d6e9009a43","To":"staging","Deployment":"something-else"}`,
						},
					},
					IsLastPage: true,
				},
			},
			found: false,
		},
	}

	for _, test := range tests {
		// add all mock pages
		for _, page := range test.pages {
			req := gock.New(testBitbucketServerURL).
				Get("/rest/api/1.0/projects/seldonio/repos/seldon-gitops/pull-requests").
				BasicAuth(testBitbucketServerCredentials.User, testBitbucketServerCredentials.Token)

			start := page.NextPageStart
			if start > 0 {
				req = req.MatchParam("start", strconv.Itoa(start))
			}

			req.Reply(http.StatusOK).
				JSON(page)
		}

		b, _ := NewBitbucketServerProvider(testBitbucketServerCredentials, prom.To)
		found, err := b.FindPromotionRequest(prom)

		g.Expect(err).ToNot(HaveOccurred())

		if test.found {
			g.Expect(found.Promotion).To(Equal(prom))
			g.Expect(found.Exists).To(BeTrue())
			g.Expect(found.ID).To(Equal(&reqID))
		} else {
			g.Expect(found).To(BeNil())
		}
	}
}
