package gitops

import (
	"net/url"
	"path"
	"strings"
)

// RepositoryDetails extracts the repository's owner and slug from the
// repository URL.
func RepositoryDetails(repoURL string) (string, string, error) {
	u, err := url.Parse(repoURL)
	if err != nil {
		return "", "", err
	}

	p := u.Path

	woSlug := path.Dir(p)
	owner := path.Base(woSlug)

	// Remove trailing ".git" if present
	slug := path.Base(p)
	slug = strings.TrimSuffix(slug, ".git")

	return owner, slug, nil
}
