package kafka

import (
	"fmt"
	"log"
	"net/url"
	"regexp"
	"strconv"

	"github.com/Shopify/sarama"
	"github.com/go-logr/logr"
	guuid "github.com/google/uuid"
	"github.com/seldonio/seldon-core/executor/api/payload"
	v1 "github.com/seldonio/seldon-core/operator/apis/machinelearning/v1"
	logf "sigs.k8s.io/controller-runtime/pkg/runtime/log"
)

type SeldonKafkaApi struct {
	KafkaConsumerGroup sarama.ConsumerGroup
	// TODO: Change back to client.SeldonApiClient
	KafkaClient    *KafkaClient
	Log            logr.Logger
	Protocol       string
	DeploymentName string
	ServerUrl      *url.URL
	predictor      *v1.PredictorSpec
}

// TODO: Change Kafka client to client.SeldonClientAPI
// TODO: Add server metrics
func NewServerKafkaApi(predictor *v1.PredictorSpec, client *KafkaClient, serverUrl *url.URL, namespace string, protocol string, deploymentName string) *SeldonKafkaApi {

	logger := logf.Log.WithName("SeldonKafkaApi")

	config := sarama.NewConfig()
	config.ClientID = deploymentName
	// TODO: Allow kafka version to be configurable
	config.Version = sarama.V2_0_0_0
	brokers := []string{fmt.Sprintf("%s:%s", serverUrl.Hostname(), serverUrl.Port())}

	groupName := fmt.Sprintf("%s-group", deploymentName)

	logger.Info("Creating Kafka Consumer", "Address", brokers)

	kafkaConsumerGroup, err := sarama.NewConsumerGroup(brokers, groupName, config)
	if err != nil {
		log.Fatalf("Failed to connect to Kafka address: %v", err)
	}

	return &SeldonKafkaApi{
		kafkaConsumerGroup,
		client,
		logger,
		protocol,
		deploymentName,
		serverUrl,
		predictor,
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
	// TODO: Explore support for multiple children for each graph node
	if len(pu.Children) > 1 {
		log.Fatal("Streaming functionality for Seldon only supports single children nodes")
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
	reStr := fmt.Sprintf("%s-(.*)-predict-output", r.DeploymentName)
	re := regexp.MustCompile(reStr)
	match := re.FindStringSubmatch(outputTopic)
	if len(match) != 2 {
		log.Fatalf("Error extracting model name from topic [%s] deployment name [%s] predictor name [%s] - resulting match [%v]", outputTopic, r.DeploymentName, r.predictor.Name, match)
	}
	modelName := match[1]
	// TODO: Add functionality to support other methods other than predict and single-child
	currModel := r.predictor.Graph
	for {
		if currModel.Name == modelName {
			if len(currModel.Children) == 0 {
				// If there's no children then return nil as we'll send topic to deployment output
				return nil
			}
			// Otherwise the next model is the first children
			// TODO: Explore support for multiple children for each graph node
			return &currModel.Children[0]
		}
		if len(currModel.Children) == 0 {
			// If there are no further children and there's no match then there is an error
			log.Fatalf("Fatal error: No model with name [%s] in Graph, from topic [%s]", modelName, outputTopic)
		}
		currModel = &currModel.Children[0]
	}
}

func (r *SeldonKafkaApi) Setup(_ sarama.ConsumerGroupSession) error   { return nil }
func (r *SeldonKafkaApi) Cleanup(_ sarama.ConsumerGroupSession) error { return nil }
func (r *SeldonKafkaApi) ConsumeClaim(sess sarama.ConsumerGroupSession, claim sarama.ConsumerGroupClaim) error {
	for msg := range claim.Messages() {
		r.Log.Info(fmt.Sprintf("Message topic:%q partition:%d offset:%d\n", msg.Topic, msg.Partition, msg.Offset))
		var model *v1.PredictiveUnit
		var meta map[string][]string
		if msg.Topic == r.KafkaClient.getDeploymentInputTopicName() {
			// Given it's the first topic we want to get the first element in the graph
			model = r.predictor.Graph
			newUUID := fmt.Sprintf("{\"ID\":\"%s\"}", guuid.New().String())
			meta = map[string][]string{payload.SeldonPUIDHeader: []string{newUUID}}
		} else {
			model = r.GetNextModelFromOutputTopic(msg.Topic)
			meta = map[string][]string{payload.SeldonPUIDHeader: []string{string(msg.Key)}}
		}

		if model != nil {
			port, err := strconv.Atoi(r.ServerUrl.Port())
			if err != nil {
				log.Fatalf("Could not convert port string to int: %v", err)
			}
			bytesPayload := payload.BytesPayload{Msg: msg.Value}
			r.KafkaClient.Predict(sess.Context(), model.Name, r.ServerUrl.Hostname(), int32(port), &bytesPayload, meta)
		} else {
			// TODO: Explore how to standardise the Predict method to work for deployment output
			topic := r.KafkaClient.getDeploymentOutputTopicName()
			r.KafkaClient.produceMessage(sess.Context(), topic, msg.Value, string(msg.Key))
		}
		// Mark messasge as read
		sess.MarkMessage(msg, "")
	}
	return nil
}
