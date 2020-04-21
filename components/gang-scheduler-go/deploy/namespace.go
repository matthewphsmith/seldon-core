package deploy

import (
	v1 "k8s.io/api/core/v1"
)

// Cluster is the response coming from the SD API
type Cluster struct {
	Namespaces []*Namespace `json:"namespaces"`
}

// TODO: We now use Namespace, but we probably should abstract this to
// "Environment" (e.g. for a multi-cluster setup).
type Namespace struct {
	v1.Namespace
}

func (n *Namespace) RepositoryURL() string {
	annotations := n.GetAnnotations()

	return annotations["git-repo"]
}

// SkipSSL will return false only if there is an explicit annotation disabling
// the skip. Otherwise, it will always default to true.
// TODO: This logic will change in the future once this is implemented in SD.
// At the moment this is only needed for tests!
func (n *Namespace) SkipSSL() bool {
	annotations := n.GetAnnotations()

	s, ok := annotations["skip-ssl"]
	if !ok {
		return true
	}

	return s != "false"
}
