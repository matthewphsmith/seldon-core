package gitops

import (
	"encoding/json"

	"github.com/mitchellh/mapstructure"
)

type BitbucketServerRequestsPage struct {
	Values        []*BitbucketServerRequest `json:"values"`
	IsLastPage    bool                      `json:"isLastPage"`
	NextPageStart int                       `json:"nextPageStart"`
}

func (p *BitbucketServerRequestsPage) UnmarshalMap(marsh map[string]interface{}) error {
	return unmarshalMap(marsh, p)
}

type BitbucketServerRequest struct {
	ID          *int                   `json:"id,omitempty"`
	Title       string                 `json:"title"`
	Description string                 `json:"description"`
	State       string                 `json:"state"`
	Open        bool                   `json:"open"`
	Closed      bool                   `json:"closed"`
	FromRef     *BitbucketServerBranch `json:"fromRef"`
	ToRef       *BitbucketServerBranch `json:"toRef"`
}

func NewBitbucketServerRequest(req *Request) *BitbucketServerRequest {
	return &BitbucketServerRequest{
		Title:       req.Title(),
		Description: req.Description(),
		State:       "OPEN",
		Open:        true,
		Closed:      false,
		FromRef:     &BitbucketServerBranch{ID: req.Branch()},
		ToRef:       &BitbucketServerBranch{ID: "master"},
	}
}

func (r *BitbucketServerRequest) DescriptionMetadata() (*RequestMetadata, error) {
	meta := &RequestMetadata{}

	err := json.Unmarshal([]byte(r.Description), &meta)
	if err != nil {
		return nil, err
	}

	return meta, nil
}

func (r *BitbucketServerRequest) MarshalMap() (map[string]interface{}, error) {
	marsh, err := marshalMap(r)
	if err != nil {
		return nil, err
	}

	// mapstructure doesn't seem to follow struct pointers
	if r.FromRef != nil {
		f, err := r.FromRef.MarshalMap()
		if err != nil {
			return nil, err
		}

		marsh["fromRef"] = f
	}

	if r.ToRef != nil {
		f, err := r.ToRef.MarshalMap()
		if err != nil {
			return nil, err
		}

		marsh["toRef"] = f
	}

	return marsh, nil
}

type BitbucketServerBranch struct {
	ID string `json:"id"`
}

func (b *BitbucketServerBranch) MarshalMap() (map[string]interface{}, error) {
	return marshalMap(b)
}

func marshalMap(input interface{}) (map[string]interface{}, error) {
	var output map[string]interface{}

	err := unmarshalMap(input, &output)
	if err != nil {
		return nil, err
	}

	return output, nil
}

func unmarshalMap(input interface{}, output interface{}) error {
	conf := &mapstructure.DecoderConfig{
		Result:  &output,
		TagName: "json",
	}

	decoder, err := mapstructure.NewDecoder(conf)
	if err != nil {
		return err
	}

	err = decoder.Decode(input)
	if err != nil {
		return err
	}

	return nil
}
