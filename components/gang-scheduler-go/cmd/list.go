package cmd

import (
	"fmt"
	"os"
	"text/tabwriter"

	"github.com/SeldonIO/seldon-deploy-cli/config"
	"github.com/SeldonIO/seldon-deploy-cli/deploy"
	"github.com/SeldonIO/seldon-deploy-cli/log"

	"github.com/spf13/cobra"
)

var listCmd = &cobra.Command{ // nolint:gochecknoglobals
	Use:   "list",
	Short: "List model list in Seldon Deploy.",
	PreRun: func(cmd *cobra.Command, args []string) {
		f := cmd.Flags()

		err := config.BindPFlags(f)
		if err != nil {
			log.Fatal(err)
		}
	},
	Run: func(cmd *cobra.Command, args []string) {
		namespace := config.GetString("namespace")

		client, err := deployClient()
		if err != nil {
			log.Fatalf("Error creating Seldon Deploy client: %s", err)
		}

		deps, err := client.GetDeployments(namespace)
		if err != nil {
			log.Fatalf("Error fetching deployments: %s", err)
		}

		if len(deps) == 0 {
			log.Infof("No model deployments found in %s namespace", namespace)
		} else {
			printDeployments(deps)
		}
	},
}

func init() { // nolint:gochecknoinits
	rootCmd.AddCommand(listCmd)

	f := listCmd.Flags()
	f.StringP("namespace", "n", "default", "Namespace to list.")
}

func printDeployments(deps []deploy.Deployment) {
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 5, ' ', tabwriter.TabIndent)
	fmt.Fprintln(w, "NAME\tKIND\tIMAGE")

	for _, dep := range deps {
		det := dep.Details()
		fmt.Fprintf(w, "%s\t%s\t%s\n", det.Name, det.Kind, det.ModelImage)
	}

	w.Flush()
}
