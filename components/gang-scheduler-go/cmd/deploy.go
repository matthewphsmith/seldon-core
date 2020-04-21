package cmd

import (
	"github.com/SeldonIO/seldon-deploy-cli/config"
	"github.com/SeldonIO/seldon-deploy-cli/deploy"
	"github.com/SeldonIO/seldon-deploy-cli/log"
	"github.com/lithammer/dedent"

	"github.com/spf13/cobra"
)

var deployCmd = &cobra.Command{ // nolint:gochecknoglobals
	Use:   "deploy [name]",
	Short: "Deploy a new version of your model.",
	Long: dedent.Dedent(`
    Deploy a new version of your model, specifying its main parameters.
    This command will always generate a new spec on each run, therefore it can
    be used both to create a new deployment and to update an existing one.
  `),
	PreRun: func(cmd *cobra.Command, args []string) {
		f := cmd.Flags()

		err := config.BindPFlags(f)
		if err != nil {
			log.Fatal(err)
		}
	},
	Args: cobra.ExactArgs(1), // nolint:gomnd
	Run: func(cmd *cobra.Command, args []string) {
		conf := config.Config()

		name := args[0]
		namespace := conf.GetString("namespace")
		modelImage := conf.GetString("model-image")
		// There is currently a bug in `viper` / `cobra` which doesn't let you
		// recover StringMapString flags as map[string]string
		// https://github.com/spf13/cobra/issues/778
		keyValList := conf.GetStringSlice("model-env")
		replicas := conf.GetInt32("replicas")

		modelEnv, err := toEnvList(keyValList)
		if err != nil {
			log.Fatalf("Error parsing --model-env flag: %s", err)
		}

		det := &deploy.DeploymentDetails{
			Namespace:  namespace,
			Name:       name,
			ModelImage: modelImage,
			ModelEnv:   modelEnv,
			Replicas:   replicas,
		}
		dep := deploy.NewDeploymentFromDetails(det)

		client, err := deployClient()
		if err != nil {
			log.Fatalf("Error creating Seldon Deploy client: %s", err)
		}

		err = client.UpdateDeployment(dep)
		if err != nil {
			log.Fatalf("Error deploying model %s: %s", det.Name, err)
		}

		log.Infof("Deployed model %s in namespace %s", det.Name, det.Namespace)
	},
}

func init() { // nolint:gochecknoinits
	rootCmd.AddCommand(deployCmd)

	f := deployCmd.Flags()
	f.StringP("namespace", "n", "default", "Namespace where to deploy the model.")
	f.StringP("model-image", "i", "", "Main model image.")
	f.StringSliceP("model-env", "e", []string{}, "Main model environment variables.")
	f.IntP("replicas", "r", 1, "Number of replicas for model deployment.")
}
