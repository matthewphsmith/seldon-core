package cmd

import (
	"fmt"

	"github.com/SeldonIO/seldon-deploy-cli/version"
	"github.com/spf13/cobra"
)

// versionCmd represents the version command
var versionCmd = &cobra.Command{ // nolint:gochecknoglobals
	Use:   "version",
	Short: "Print the version number of generated code example",
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("Build Date:", version.BuildDate)
		fmt.Println("Git Commit:", version.GitCommit)
		fmt.Println("Version:", version.Version)
		fmt.Println("Go Version:", version.GoVersion)
		fmt.Println("OS / Arch:", version.OsArch)
	},
}

func init() { // nolint:gochecknoinits
	rootCmd.AddCommand(versionCmd)
}
