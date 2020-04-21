package config

import (
	"time"

	"github.com/spf13/pflag"
	"github.com/spf13/viper"
)

// Provider defines a set of read-only methods for accessing the application
// configuration params as defined in one of the config files.
type Provider interface {
	ConfigFileUsed() string
	Get(key string) interface{}
	GetBool(key string) bool
	GetDuration(key string) time.Duration
	GetFloat64(key string) float64
	GetInt(key string) int
	GetInt32(key string) int32
	GetInt64(key string) int64
	GetSizeInBytes(key string) uint
	GetString(key string) string
	GetStringMap(key string) map[string]interface{}
	GetStringMapString(key string) map[string]string
	GetStringMapStringSlice(key string) map[string][]string
	GetStringSlice(key string) []string
	GetTime(key string) time.Time
	InConfig(key string) bool
	IsSet(key string) bool

	BindPFlags(*pflag.FlagSet) error
}

var defaultConfig *viper.Viper // nolint:gochecknoglobals

// Config returns a default config providers
func Config() Provider {
	return defaultConfig
}

// LoadConfigProvider returns a configured viper instance
func LoadConfigProvider(appName string) Provider {
	return readViperConfig(appName)
}

func init() { // nolint:gochecknoinits
	defaultConfig = readViperConfig("SELDON-DEPLOY-CLI")
}

func readViperConfig(appName string) *viper.Viper {
	v := viper.New()
	v.SetEnvPrefix(appName)
	v.AutomaticEnv()

	// global defaults
	v.SetDefault("loglevel", "debug")

	return v
}

// GetString package-level convenience method.
func GetString(key string) string {
	return defaultConfig.GetString(key)
}

// GetStringMapString package-level convenience method.
func GetStringMapString(key string) map[string]string {
	return defaultConfig.GetStringMapString(key)
}

// GetInt32 package-level convenience method.
func GetInt32(key string) int32 {
	return defaultConfig.GetInt32(key)
}

// BindPFlags package-level convenience method.
func BindPFlags(flags *pflag.FlagSet) error {
	return defaultConfig.BindPFlags(flags)
}
