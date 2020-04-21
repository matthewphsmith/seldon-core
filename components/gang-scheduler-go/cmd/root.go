package cmd

import (
	"github.com/SeldonIO/seldon-deploy-cli/config"
	"github.com/SeldonIO/seldon-deploy-cli/log"
	"github.com/lithammer/dedent"
	"github.com/spf13/cobra"
)

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{ // nolint:gochecknoglobals
	Use:   "sd",
	Short: "sd allows to control your machine learning resources in Seldon Deploy.",
	Long: dedent.Dedent(`
    sd is a command line interface which allows you to access Seldon Deploy
    programmatically.
    Complete documentation is available at https://deploy-master.seldon.io/docs/cli/
  `),
}

// Execute adds all child commands to the root command and sets flags appropriately.
// This is called by main.main(). It only needs to happen once to the rootCmd.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		log.Fatal(err)
	}
}

func init() { // nolint:gochecknoinits
	pf := rootCmd.PersistentFlags()
	pf.StringP("server", "s", "", "Seldon Deploy server URL.")
	pf.StringP("user", "u", "", "Seldon Deploy user.")
	pf.StringP("password", "p", "", "Seldon Deploy password.")

	err := config.BindPFlags(pf)
	if err != nil {
		log.Fatal(err)
	}
}
