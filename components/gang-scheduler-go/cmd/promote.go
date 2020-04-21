package cmd

import (
	"github.com/SeldonIO/seldon-deploy-cli/config"
	"github.com/SeldonIO/seldon-deploy-cli/deploy"
	"github.com/SeldonIO/seldon-deploy-cli/gitops"
	"github.com/SeldonIO/seldon-deploy-cli/log"
	"github.com/lithammer/dedent"

	"github.com/spf13/cobra"
)

var promoteCmd = &cobra.Command{ // nolint:gochecknoglobals
	Use:   "promote [name]",
	Short: "Promote your model between environments.",
	Long: dedent.Dedent(`
    Submit a promotion request for your model between environments.
    If there is an existing request for the same model, this command
    will update it.
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
		name := args[0]
		fromNamespace := config.GetString("from")
		toNamespace := config.GetString("to")
		gitUser := config.GetString("git-user")
		gitToken := config.GetString("git-token")
		promAuthor := config.GetString("promotion-author")
		promEmail := config.GetString("promotion-email")

		// Find deployment and target environment
		client, err := deployClient()
		if err != nil {
			log.Fatalf("Error creating Seldon Deploy client: %s", err)
		}

		dep, err := client.GetDeployment(fromNamespace, name)
		if err != nil {
			log.Fatalf("Error finding model %s in namespace %s: %s", name, fromNamespace, err)
		}

		toNS, err := client.GetNamespace(toNamespace)
		if err != nil {
			log.Fatalf("Error finding destination namespace %s: %s", toNamespace, err)
		}

		// Mutate deployment
		det := &deploy.DeploymentDetails{
			Namespace: toNamespace,
			Name:      name,
		}
		updated, err := dep.
			WithoutStatus().
			WithDetails(det)
		if err != nil {
			log.Fatalf("Error updating details of model %s: %s", name, err)
		}

		// Clone repo and request promotion
		creds := &gitops.Credentials{
			User:  gitUser,
			Token: gitToken,
		}

		r, err := gitops.CloneGitOpsRepository(creds, toNS)
		if err != nil {
			log.Fatalf("Error cloning GitOps repository: %s", err)
		}
		defer r.Close()

		prom := &gitops.Promotion{
			Deployment: updated,
			To:         toNS,
			Author:     promAuthor,
			Email:      promEmail,
		}

		err = r.RequestPromotion(prom)
		if err != nil {
			log.Fatalf("Error requesting promotion: %s", err)
		}

		log.Infof(
			"Requested promotion to %s for model %s",
			toNamespace,
			name,
		)
	},
}

func init() { // nolint:gochecknoinits
	rootCmd.AddCommand(promoteCmd)

	f := promoteCmd.Flags()
	f.StringP("from", "f", "", "Current environment of the model.")
	f.StringP("to", "t", "", "Environment where we want to promote our model to.")
	f.String("git-user", "", "Git user of the target GitOps repository.")
	f.String("git-token", "", "Git token of the target GitOps repository.")
	f.String("promotion-author", "", "Name of the author of the promotion request.")
	f.String("promotion-email", "", "Email of the author of the promotion request.")
}
