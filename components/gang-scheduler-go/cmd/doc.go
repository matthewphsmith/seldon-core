package cmd

import (
	"fmt"
	"path"
	"path/filepath"
	"strings"
	"time"

	"github.com/SeldonIO/seldon-deploy-cli/config"
	"github.com/SeldonIO/seldon-deploy-cli/log"

	"github.com/lithammer/dedent"
	"github.com/spf13/cobra"
	"github.com/spf13/cobra/doc"
)

const (
	docOutputMarkdown = "markdown"
	docOutputHugo     = "hugo"
	docFrontMatter    = `
    ---
    date: %s
    title: %s
    slug: %s
    url: %s
    ---
  `
)

// nolint:gochecknoglobals
var validDocOutput = fmt.Sprintf("%s|%s", docOutputMarkdown, docOutputHugo)

var docCmd = &cobra.Command{ // nolint:gochecknoglobals
	Use:   "doc [directory]",
	Short: "Generate the documentation for sd.",
	Args:  cobra.ExactArgs(1), // nolint:gomnd
	PreRun: func(cmd *cobra.Command, args []string) {
		f := cmd.Flags()

		err := config.BindPFlags(f)
		if err != nil {
			log.Fatal(err)
		}

		output := config.GetString("output")
		if output != docOutputMarkdown && output != docOutputHugo {
			log.Fatalf("Invalid output %s. Valid ones are %s", output, validDocOutput)
		}
	},
	Run: func(cmd *cobra.Command, args []string) {
		directory := args[0]
		output := config.GetString("output")

		var err error
		if output == docOutputMarkdown {
			err = doc.GenMarkdownTree(rootCmd, directory)
		} else if output == docOutputHugo {
			err = doc.GenMarkdownTreeCustom(rootCmd, directory, filePrepender, linkHandler)
		}

		if err != nil {
			log.Fatal(err)
		}

		log.Infof("Generated documentation in %s", directory)
	},
}

func filePrepender(fpath string) string {
	now := time.Now().Format(time.RFC3339)
	slug := getSlug(fpath)
	title := strings.Replace(slug, "-", " ", -1)
	url := linkHandler(fpath)

	fm := fmt.Sprintf(
		dedent.Dedent(docFrontMatter),
		now,
		title,
		slug,
		url,
	)

	// Trim initial newline and add one at the end
	fm = strings.TrimLeft(fm, "\n")
	fm += "\n"

	return fm
}

func linkHandler(fname string) string {
	slug := getSlug(fname)
	return fmt.Sprintf("/docs/tour-of-seldon-deploy/cli/%s/", slug)
}

func getSlug(fpath string) string {
	fname := filepath.Base(fpath)
	fext := path.Ext(fname)
	slug := strings.TrimSuffix(fname, fext)

	return strings.Replace(slug, "_", "-", -1)
}

func init() { // nolint:gochecknoinits
	rootCmd.AddCommand(docCmd)

	f := docCmd.Flags()

	outputUsage := fmt.Sprintf("Output format of the docs. One of: %s.", validDocOutput)
	f.StringP("output", "o", "markdown", outputUsage)
}
