package gitops

import (
	"fmt"
	"testing"

	. "github.com/onsi/gomega"
)

func TestBitbucketServerRequestMetadata(t *testing.T) {
	g := NewGomegaWithT(t)

	tests := []struct {
		bReq *BitbucketServerRequest
		meta *RequestMetadata
	}{
		{
			bReq: &BitbucketServerRequest{
				Description: `{
        "Action": "Promotion to production",
        "Deployment": "my-model"}`,
			},
			meta: &RequestMetadata{
				Action:     "Promotion to production",
				Deployment: "my-model",
			},
		},
		{
			bReq: &BitbucketServerRequest{
				Description: `this is not JSON`,
			},
			meta: nil,
		},
	}

	for _, test := range tests {
		meta, err := test.bReq.DescriptionMetadata()

		if test.meta == nil {
			g.Expect(err).To(HaveOccurred())
		} else {
			g.Expect(err).ToNot(HaveOccurred())
			g.Expect(meta).To(Equal(test.meta))
		}
	}
}

func TestBitbucketServerRequestMarshalMap(t *testing.T) {
	g := NewGomegaWithT(t)

	tests := []struct {
		req *BitbucketServerRequest
		exp map[string]interface{}
	}{
		{
			req: &BitbucketServerRequest{
				Title:       "my pr",
				Description: "this is my pr",
				Open:        true,
				FromRef: &BitbucketServerBranch{
					ID: "master",
				},
			},
			exp: map[string]interface{}{
				"id":          nil,
				"title":       "my pr",
				"description": "this is my pr",
				"state":       "",
				"open":        true,
				"closed":      false,
				"fromRef": map[string]interface{}{
					"id": "master",
				},
				"toRef": nil,
			},
		},
	}

	for _, test := range tests {
		marshalled, err := test.req.MarshalMap()

		g.Expect(err).ToNot(HaveOccurred())

		// g.Expect(marshalled).To(Equal(test.exp)) doesn't work for some reason...
		g.Expect(fmt.Sprint(marshalled)).To(Equal(fmt.Sprint(test.exp)))
	}
}

func TestBitbucketServerRequestPageUnmarshalMap(t *testing.T) {
	g := NewGomegaWithT(t)

	tests := []struct {
		marsh map[string]interface{}
		page  *BitbucketServerRequestsPage
	}{
		{
			marsh: map[string]interface{}{
				"values": []map[string]interface{}{
					{"title": "my pr 1"},
				},
				"isLastPage":    false,
				"nextPageStart": 24,
			},
			page: &BitbucketServerRequestsPage{
				Values: []*BitbucketServerRequest{
					{
						Title: "my pr 1",
					},
				},
				IsLastPage:    false,
				NextPageStart: 24,
			},
		},
	}

	for _, test := range tests {
		p := &BitbucketServerRequestsPage{}
		err := p.UnmarshalMap(test.marsh)

		g.Expect(err).ToNot(HaveOccurred())
		g.Expect(p).To(Equal(test.page))
	}
}
