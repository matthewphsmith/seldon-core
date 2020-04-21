package deploy

import (
	"fmt"
	"net/http"
	"net/http/cookiejar"
	"net/url"
	"testing"

	. "github.com/onsi/gomega"
	"gopkg.in/h2non/gock.v1"
)

const (
	testSessionCookie = "my-auth-cookie-value"
	testUser          = "admin@seldon.io"
	testPassword      = "12341234"
)

func authFixture() *Authenticator {
	auth, _ := NewAuthenticator(testServerURL, testUser, testPassword)
	auth.session = testSessionCookie

	return auth
}

func TestAuthenticateClient(t *testing.T) {
	g := NewGomegaWithT(t)

	jar, _ := cookiejar.New(nil)
	tests := []struct {
		cli  *http.Client
		auth *Authenticator
	}{
		{
			cli:  &http.Client{},
			auth: authFixture(),
		},
		{
			cli:  &http.Client{Jar: jar},
			auth: authFixture(),
		},
	}

	for _, test := range tests {
		cli, err := test.auth.AuthenticateClient(test.cli)

		g.Expect(err).ToNot(HaveOccurred())
		g.Expect(cli.Jar).ToNot(BeNil())

		s, _ := url.Parse(test.auth.Server)
		cookies := cli.Jar.Cookies(s)
		cookie := test.auth.cookie()
		g.Expect(cookies).To(ContainElement(cookie))
	}
}

func TestAuthPath(t *testing.T) {
	g := NewGomegaWithT(t)

	defer gock.Off()

	expected := "/dex/auth/local"

	auth := authFixture()

	u, _ := url.Parse(testServerURL)
	gock.New(auth.host).
		Get(u.Path).
		Reply(http.StatusFound).
		AddHeader("Location", "/dex/auth")

	gock.New(auth.host).
		Get("/dex/auth").
		Reply(http.StatusFound).
		AddHeader("Location", expected)

	authPath, err := auth.authPath()

	g.Expect(err).ToNot(HaveOccurred())
	g.Expect(authPath).To(Equal(expected))
}

func TestSubmitAuth(t *testing.T) {
	g := NewGomegaWithT(t)

	defer gock.Off()

	authPath := "/dex/auth/local"
	expected := "/login/oidc"

	auth := authFixture()

	gock.New(auth.host).
		Post(authPath).
		BodyString(auth.authPayload().Encode()).
		MatchType("url").
		Reply(http.StatusSeeOther).
		AddHeader("Location", "/dex/approval")

	gock.New(auth.host).
		Get("/dex/approval").
		Reply(http.StatusSeeOther).
		AddHeader("Location", expected)

	successPath, err := auth.submitAuth(authPath)

	g.Expect(err).ToNot(HaveOccurred())
	g.Expect(successPath).To(Equal(expected))
}

func TestSessionCookie(t *testing.T) {
	g := NewGomegaWithT(t)

	defer gock.Off()

	auth := authFixture()

	successPath := "/login/oidc"
	gock.New(auth.host).
		Get(successPath).
		Reply(http.StatusFound).
		AddHeader("Location", auth.Server).
		AddHeader(
			"Set-Cookie",
			fmt.Sprintf("%s=%s", sessionCookieName, testSessionCookie),
		)

	cookie, err := auth.sessionCookie(successPath)

	g.Expect(err).ToNot(HaveOccurred())
	g.Expect(cookie.Name).To(Equal(sessionCookieName))
	g.Expect(cookie.Value).To(Equal(testSessionCookie))
}
