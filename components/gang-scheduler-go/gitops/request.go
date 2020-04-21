package gitops

import (
	"encoding/json"
	"fmt"

	"github.com/google/uuid"
)

type Request struct {
	ID        *uuid.UUID
	Promotion *Promotion
	Exists    bool
}

// RequestMetadata represents the minimum info about a promotion Request, which
// gets serialised on the PR's body.
type RequestMetadata struct {
	ID         *uuid.UUID `json:"ID,omitempty"`
	Action     string     `json:"Action"`
	To         string     `json:"To"`
	Deployment string     `json:"Deployment"`
}

// NewRequestMetadata will create the metadata section of a promotion request
// from a given promotion. Note that, at this point the request ID will be
// empty.
func NewRequestMetadata(prom *Promotion) *RequestMetadata {
	dets := prom.Deployment.Details()

	return &RequestMetadata{
		Action:     fmt.Sprintf("Promotion to %s", prom.To.Name),
		To:         prom.To.Name,
		Deployment: dets.Name,
	}
}

func (m *RequestMetadata) IsEqual(b *RequestMetadata) bool {
	// If ID is present, consider them equal if they are the same
	if m.ID != nil && b.ID != nil {
		return m.ID == b.ID
	}

	// Consider both requests equal if target namespace and deployment name is
	// the same
	return m.To == b.To && m.Deployment == b.Deployment
}

func (r *Request) Branch() string {
	if r.ID == nil {
		u := uuid.New()
		r.ID = &u
	}

	return fmt.Sprintf("promotion/%s", r.ID)
}

func (r *Request) Title() string {
	dets := r.Promotion.Deployment.Details()
	return fmt.Sprintf("Promotion Request for %s", dets.Name)
}

func (r *Request) Metadata() *RequestMetadata {
	meta := NewRequestMetadata(r.Promotion)
	meta.ID = r.ID

	return meta
}

func (r *Request) Description() string {
	meta := r.Metadata()

	desc, err := json.Marshal(meta)
	if err != nil {
		return ""
	}

	return string(desc)
}
