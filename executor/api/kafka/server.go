package kafka

import (
	"context"
	"fmt"
	"log"
	"net/url"
	"strconv"

	"github.com/Shopify/sarama"
	"github.com/go-logr/logr"
	"github.com/seldonio/seldon-core/executor/api/client"
	"github.com/seldonio/seldon-core/executor/api/metric"
	"github.com/seldonio/seldon-core/executor/api/payload"
	v1 "github.com/seldonio/seldon-core/operator/apis/machinelearning/v1"
	logf "sigs.k8s.io/controller-runtime/pkg/runtime/log"
)

type SeldonKafkaApi struct {
	KafkaConsumerGroup sarama.ConsumerGroup
	// TODO: Change back to client.SeldonApiClient
	KafkaClient    KafkaClient
	Log            logr.Logger
	ProbesOnly     bool
	Protocol       string
	DeploymentName string
	ServerUrl      *url.URL
	predictor      *v1.PredictorSpec
	metrics        *metric.ServerMetrics
	prometheusPath string
}

func NewServerKafkaApi(predictor *v1.PredictorSpec, client client.SeldonApiClient, probesOnly bool, serverUrl *url.URL, namespace string, protocol string, deploymentName string, prometheusPath string) *SeldonKafkaApi {

	var serverMetrics *metric.ServerMetrics
	if !probesOnly {
		serverMetrics = metric.NewServerMetrics(predictor, deploymentName)
	}

	config := sarama.NewConfig()
	config.ClientID = deploymentName
	brokers := []string{serverUrl.String()}

	groupName := fmt.Sprintf("%s-group", deploymentName)

	kafkaConsumerGroup, err := sarama.NewConsumerGroup(brokers, groupName, config)
	if err != nil {
		log.Fatalf("Failed to connect to Kafka address: %v", err)
	}

	return &SeldonKafkaApi{
		kafkaConsumerGroup,
		client,
		logf.Log.WithName("SeldonKafkaApi"),
		probesOnly,
		protocol,
		deploymentName,
		serverUrl,
		predictor,
		serverMetrics,
		prometheusPath,
	}
}

func (r *SeldonKafkaApi) ModelNameToInputTopic(modelName string) string {
	return fmt.Sprintf("%s-%s-input", r.DeploymentName, modelName)
}

func (r *SeldonKafkaApi) RecursiveAppendTopicNamesToConsume(pu *v1.PredictiveUnit, topics *[]string) {
	topicName := r.KafkaClient.getModelOutputTopicName("predict", pu.Name)
	*topics = append(*topics, topicName)
	// TODO: Explore support for other methods beyond Predict
	// Currently only models (predict) functions supported
	// If any of the models contain non-predict models it would fail
	if *pu.Type != v1.MODEL {
		log.Fatal("Streaming functionality for seldon only supports MODEL predictors, and does not support other seldon deployment component types for now.")
	}
	for i := 0; i < len(pu.Children); i++ {
		r.RecursiveAppendTopicNamesToConsume(&pu.Children[i], topics)
	}
}

func (r *SeldonKafkaApi) GetTopicNamesToConsume() []string {
	topicsToConsume := []string{}
	pu := r.predictor.Graph
	// We first append the deployment input topic
	topicsToConsume = append(topicsToConsume, r.KafkaClient.getDeploymentInputTopicName())
	r.RecursiveAppendTopicNamesToConsume(pu, &topicsToConsume)
	return topicsToConsume
}

func (r *SeldonKafkaApi) GetNextModelFromOutputTopic(outputTopic string) *v1.PredictiveUnit {
	// TODO: Make this work with other methods other than predict
	reStr := fmt.Sprintf("%s-%s-(.*)-predict-output", r.DeploymentName, r.predictor.Name)
	re := regexp.MustCompile(reStr)
}

func (r *SeldonKafkaApi) Setup(_ sarama.ConsumerGroupSession) error   { return nil }
func (r *SeldonKafkaApi) Cleanup(_ sarama.ConsumerGroupSession) error { return nil }
func (r *SeldonKafkaApi) ConsumeClaim(sess sarama.ConsumerGroupSession, claim sarama.ConsumerGroupClaim) error {
	for msg := range claim.Messages() {
		r.Log.Info("Message topic:%q partition:%d offset:%d\n", msg.Topic, msg.Partition, msg.Offset)
		if msg.Topic == r.KafkaClient.getDeploymentInputTopicName() {
			// Given it's the first topic we want to get the first element in the graph
			firstModel := r.predictor.Graph
			port, err := strconv.Atoi(r.ServerUrl.Port())
			if err != nil {
				log.Fatalf("Could not convert port string to int: %v", err)
			}
			bytesPayload := payload.BytesPayload{Msg: msg.Value}
			meta := map[string][]string{payload.SeldonPUIDHeader: []string{string(msg.Key)}}
			r.KafkaClient.Predict(sess.Context(), firstModel.Name, r.ServerUrl.Hostname(), int32(port), &bytesPayload, meta)
		} else {
			nextModel := r.GetNextModelFromOutputTopic(msg.Topic)
			if nextModel != nil {
				port, err := strconv.Atoi(r.ServerUrl.Port())
				if err != nil {
					log.Fatalf("Could not convert port string to int: %v", err)
				}
				bytesPayload := payload.BytesPayload{Msg: msg.Value}
				meta := map[string][]string{payload.SeldonPUIDHeader: []string{string(msg.Key)}}
				r.KafkaClient.Predict(sess.Context(), nextModel.Name, r.ServerUrl.Hostname(), int32(port), &bytesPayload, meta)
			}
			else {
				// If there is no next model then we send the final output to the deploymen output topic
			}
		}
		// Mark messasge as read
		sess.MarkMessage(msg, "")
	}
	return nil
}

func runKafkaServerApi(r *SeldonKafkaApi) {
	topics := r.GetTopicNamesToConsume()
	ctx := context.Background()
	for {
		err := r.KafkaConsumerGroup.Consume(ctx, topics, r)
		if err != nil {
			panic(err)
		}
	}
}
