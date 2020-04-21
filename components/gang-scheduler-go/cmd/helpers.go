package cmd

import (
	"fmt"
	"strings"

	"github.com/SeldonIO/seldon-deploy-cli/config"
	"github.com/SeldonIO/seldon-deploy-cli/deploy"

	v1 "k8s.io/api/core/v1"
)

func toEnvList(flags []string) ([]v1.EnvVar, error) {
	envList := []v1.EnvVar{}

	for _, keyVal := range flags {
		items := strings.Split(keyVal, "=")
		if len(items) != 2 { // nolint:gomnd
			return nil, fmt.Errorf("invalid flag value %s", keyVal)
		}

		envVar := v1.EnvVar{Name: items[0], Value: items[1]}
		envList = append(envList, envVar)
	}

	return envList, nil
}

func deployClient() (*deploy.Client, error) {
	server := config.GetString("server")
	user := config.GetString("user")
	password := config.GetString("password")

	auth, err := deploy.NewAuthenticator(server, user, password)
	if err != nil {
		return nil, err
	}

	err = auth.Authenticate()
	if err != nil {
		return nil, err
	}

	return deploy.NewClient(server, auth)
}
