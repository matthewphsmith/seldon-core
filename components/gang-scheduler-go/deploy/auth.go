package deploy

import (
	"fmt"
	"net/http"
	"net/http/cookiejar"
	"net/url"
)

const (
	sessionCookieName = "authservice_session"
)

// Authenticator which will obtain the credentials for a given session and will
// provide a method to authenticate requests by adding any necessary
// credentials. This may involve in the future to an interface which will allow
// us to support different authentication mechanisms (e.g. API tokens, etc.)
type Authenticator struct {
	Server string

	session string

	// Fields required for initial auth flow
	host     string
	cli      *http.Client
	user     string
	password string
}

// NewAuthenticator uses the given credentials to instantiate an authenticator.
func NewAuthenticator(server string, user string, password string) (*Authenticator, error) {
	// Disable redirects
	cli := &http.Client{
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}

	u, err := url.Parse(server)
	if err != nil {
		return nil, err
	}

	host := fmt.Sprintf("%s://%s", u.Scheme, u.Host)
	auth := &Authenticator{
		Server:   server,
		host:     host,
		cli:      cli,
		user:     user,
		password: password,
	}

	return auth, nil
}

// Authenticate mimics the steps described in the following GH issue to obtain
// a session cookie:
// https://github.com/kubeflow/kfctl/issues/140#issuecomment-578837304
func (a *Authenticator) Authenticate() error {
	authPath, err := a.authPath()
	if err != nil {
		return err
	}

	successPath, err := a.submitAuth(authPath)
	if err != nil {
		return err
	}

	sessionCookie, err := a.sessionCookie(successPath)
	if err != nil {
		return err
	}

	a.session = sessionCookie.Value

	return nil
}

// authPath will obtain the URL path to submit our authentication payload.
func (a *Authenticator) authPath() (string, error) {
	// Since we require authentication, this should redirect to the OIDC provider
	res, err := a.cli.Get(a.Server)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusFound {
		return "", authError()
	}

	// Sending another GET request to the OIDC provider should redirect us to the
	// default auth URL
	oidcPath := res.Header.Get("Location")

	res, err = a.cli.Get(a.endpoint(oidcPath))
	if err != nil {
		return "", err
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusFound {
		return "", authError()
	}

	// Lastly, the OIDC provider will redirect us to the login form. We can also
	// use this URL to send a POST request with the user credentials.
	// TODO: How does this work with multiple Dex providers?
	return res.Header.Get("Location"), nil
}

// submitAuth will submit the auth payload and will return the final success
// URL path which tries to redirect us to Seldon Deploy and sets the session
// cookie.
func (a *Authenticator) submitAuth(authPath string) (string, error) {
	authPayload := a.authPayload()

	// We can use the auth URL to send a POST request with the credentials
	res, err := a.cli.PostForm(a.endpoint(authPath), *authPayload)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusSeeOther {
		return "", authError()
	}

	// Next, the response should contain a redirect to obtain the OIDC login URL
	loginPath := res.Header.Get("Location")

	res, err = a.cli.Get(a.endpoint(loginPath))
	if err != nil {
		return "", err
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusSeeOther {
		return "", authError()
	}

	// Lastly, the login URL should attempt to redirect us to Seldon Deploy, setting
	// the session cookie
	return res.Header.Get("Location"), nil
}

func (a *Authenticator) authPayload() *url.Values {
	payload := &url.Values{}
	payload.Set("login", a.user)
	payload.Set("password", a.password)

	return payload
}

// sessionCookie will follow the success URL to obtain the session cookie
func (a *Authenticator) sessionCookie(successPath string) (*http.Cookie, error) {
	res, err := a.cli.Get(a.endpoint(successPath))
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusFound {
		return nil, authError()
	}

	// We can extract the cookie from the last redirect
	cookies := res.Cookies()
	for _, cookie := range cookies {
		if cookie.Name == sessionCookieName {
			return cookie, nil
		}
	}

	return nil, authError()
}

func (a *Authenticator) endpoint(path string) string {
	return fmt.Sprintf("%s%s", a.host, path)
}

// Authenticate a request by adding any necessary credentials.
func (a *Authenticator) AuthenticateClient(cli *http.Client) (*http.Client, error) {
	if cli.Jar == nil {
		j, err := cookiejar.New(nil)
		if err != nil {
			return nil, err
		}

		cli.Jar = j
	}

	c := a.cookie()

	s, err := url.Parse(a.Server)
	if err != nil {
		return nil, err
	}

	cli.Jar.SetCookies(s, []*http.Cookie{c})

	return cli, nil
}

func (a *Authenticator) cookie() *http.Cookie {
	return &http.Cookie{
		Name:  sessionCookieName,
		Value: a.session,
	}
}
