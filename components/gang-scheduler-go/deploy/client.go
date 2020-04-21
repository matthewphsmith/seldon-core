package deploy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"

	"github.com/SeldonIO/seldon-deploy-cli/log"
)

// Client allows to interact with the Seldon Deploy API.
type Client struct {
	Server        string
	Authenticator *Authenticator

	cli *http.Client
}

// NewClient instantiates a client with all the required authentication steps.
func NewClient(server string, auth *Authenticator) (*Client, error) {
	cli := &http.Client{}

	cli, err := auth.AuthenticateClient(cli)
	if err != nil {
		return nil, err
	}

	client := &Client{Server: server, Authenticator: auth, cli: cli}

	return client, nil
}

// GetNamespaces retrieves the list of namespaces
func (c *Client) GetNamespaces() ([]*Namespace, error) {
	endpoint := fmt.Sprintf("%s/api/cluster", c.Server)

	// nolint:gosec
	resp, err := c.cli.Get(endpoint)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	err = checkResponseError(resp)
	if err != nil {
		return nil, err
	}

	cluster := &Cluster{}
	dec := json.NewDecoder(resp.Body)

	err = dec.Decode(&cluster)
	if err != nil {
		return nil, err
	}

	return cluster.Namespaces, nil
}

// GetNamespace fetches a single namespace information
func (c *Client) GetNamespace(namespace string) (*Namespace, error) {
	namespaces, err := c.GetNamespaces()
	if err != nil {
		return nil, err
	}

	for _, ns := range namespaces {
		if ns.Name == namespace {
			return ns, nil
		}
	}

	return nil, namespaceNotFoundError(namespace)
}

// GetDeployments allows to get a list of deployments.
func (c *Client) GetDeployments(namespace string) ([]Deployment, error) {
	endpoint := c.deploymentsEndpoint()

	q := endpoint.Query()
	q.Set("namespace", namespace)
	endpoint.RawQuery = q.Encode()

	resp, err := c.cli.Get(endpoint.String())
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	err = checkResponseError(resp)
	if err != nil {
		return nil, err
	}

	// TODO: Support list with variable types (i.e. SeldonDeployment and
	// KFServing)
	sdeps := []*SeldonDeployment{}
	dec := json.NewDecoder(resp.Body)

	err = dec.Decode(&sdeps)
	if err != nil {
		return nil, err
	}

	// Cast back into generic Deployment
	// TODO: Remove once we support list with variable types.
	deps := []Deployment{}
	for _, sdep := range sdeps {
		deps = append(deps, sdep)
	}

	return deps, nil
}

// GetDeployment fetches a single deployment information
func (c *Client) GetDeployment(namespace string, name string) (Deployment, error) {
	deps, err := c.GetDeployments(namespace)
	if err != nil {
		return nil, err
	}

	// There isn't an endpoint to fetch a single deployment yet, so we need to
	// find it on the entire list for now.
	for _, dep := range deps {
		det := dep.Details()
		if det.Name == name {
			return dep, nil
		}
	}

	return nil, deploymentNotFoundError(namespace, name)
}

// UpdateDeployment allows to update an existing deployment.
func (c *Client) UpdateDeployment(dep Deployment) error {
	endpoint := c.deploymentsEndpoint()

	det := dep.Details()
	q := endpoint.Query()
	q.Set("type", det.Kind)
	endpoint.RawQuery = q.Encode()

	payload, contentType, err := c.deploymentPayload(dep)
	if err != nil {
		return err
	}

	req, err := http.NewRequest(http.MethodPut, endpoint.String(), payload)
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", contentType)

	res, err := c.cli.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()

	err = checkResponseError(res)
	if err != nil {
		return err
	}

	return nil
}

func (c *Client) deploymentsEndpoint() *url.URL {
	raw := fmt.Sprintf("%s/api/deployments", c.Server)

	u, err := url.Parse(raw)
	if err != nil {
		log.Fatalf("Failed parsing deployments URL: %s", err)
	}

	return u
}

func (c *Client) deploymentPayload(dep Deployment) (io.Reader, string, error) {
	payload := &bytes.Buffer{}
	writer := multipart.NewWriter(payload)

	part, err := writer.CreateFormField("deployment")
	if err != nil {
		return nil, "", err
	}

	enc := json.NewEncoder(part)

	err = enc.Encode(dep)
	if err != nil {
		return nil, "", err
	}

	contentType := writer.FormDataContentType()

	writer.Close()

	return payload, contentType, nil
}
