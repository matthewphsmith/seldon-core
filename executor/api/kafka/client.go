package kafka

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"

	"github.com/Shopify/sarama"
	"github.com/go-logr/logr"
	"github.com/golang/protobuf/jsonpb"
	"github.com/seldonio/seldon-core/executor/api/grpc/seldon/proto"
	"github.com/seldonio/seldon-core/executor/api/metric"
	"github.com/seldonio/seldon-core/executor/api/payload"
	v1 "github.com/seldonio/seldon-core/operator/apis/machinelearning/v1"
	logf "sigs.k8s.io/controller-runtime/pkg/runtime/log"
)

const (
	ContentTypeJSON = "application/json"
)

type KafkaClient struct {
	kafkaProducer  sarama.SyncProducer
	Log            logr.Logger
	Protocol       string
	DeploymentName string
	predictor      *v1.PredictorSpec
	metrics        *metric.ClientMetrics
}

func (smc *KafkaClient) CreateErrorPayload(err error) payload.SeldonPayload {
	respFailed := proto.SeldonMessage{Status: &proto.Status{Code: http.StatusInternalServerError, Info: err.Error()}}
	m := jsonpb.Marshaler{}
	jStr, _ := m.MarshalToString(&respFailed)
	res := payload.BytesPayload{Msg: []byte(jStr)}
	return &res
}

func (smc *KafkaClient) Marshall(w io.Writer, msg payload.SeldonPayload) error {
	_, err := w.Write(msg.GetPayload().([]byte))
	return err
}

func (smc *KafkaClient) Unmarshall(msg []byte) (payload.SeldonPayload, error) {
	reqPayload := payload.BytesPayload{Msg: msg, ContentType: ContentTypeJSON}
	return &reqPayload, nil
}

type BytesKafkaClientOption func(client *KafkaClient)

// TODO: Change the return type into client.SeldonApiClient
func NewKafkaClient(serverUrl *url.URL, protocol string, deploymentName string, predictor *v1.PredictorSpec, options ...BytesKafkaClientOption) *KafkaClient {

	logger := logf.Log.WithName("KafkaClient")
	// TODO: Add sarama logger to the Kafka Client logger to provide more details on info
	// sarama.Logger = log.New(os.Stdout, "[sarama] ", log.LstdFlags)

	kafkaConfig := sarama.NewConfig()
	kafkaConfig.Producer.Retry.Max = 5
	kafkaConfig.ClientID = deploymentName
	kafkaConfig.Producer.RequiredAcks = sarama.WaitForAll
	kafkaConfig.Producer.Return.Successes = true

	brokers := []string{fmt.Sprintf("%s:%s", serverUrl.Hostname(), serverUrl.Port())}

	logger.Info("Creating Kafka Producer", "url", strings.Join(brokers, ", "))
	producer, err := sarama.NewSyncProducer(brokers, kafkaConfig)
	if err != nil {
		log.Fatalf("Failed to connect to Kafka address [%s] with error [%v]", brokers, err)
	}

	client := KafkaClient{
		producer,
		logger,
		protocol,
		deploymentName,
		predictor,
		metric.NewClientMetrics(predictor, deploymentName, ""),
	}

	for i := range options {
		options[i](&client)
	}

	return &client
}

func (smc *KafkaClient) getDeploymentOutputTopicName() string {
	return fmt.Sprintf("%s-output", smc.DeploymentName)
}

func (smc *KafkaClient) getDeploymentInputTopicName() string {
	return fmt.Sprintf("%s-input", smc.DeploymentName)
}

func (smc *KafkaClient) getModelOutputTopicName(method string, modelName string) string {
	return fmt.Sprintf("%s-%s-%s-output", smc.DeploymentName, modelName, method)
}

func (smc *KafkaClient) getModelInputTopicName(method string, modelName string) string {
	return fmt.Sprintf("%s-%s-%s-input", smc.DeploymentName, modelName, method)
}

func (smc *KafkaClient) produceMessage(ctx context.Context, topic string, bytesMessage []byte, uuid string) error {
	message := &sarama.ProducerMessage{
		Topic: topic,
		Key:   sarama.StringEncoder(uuid),
		Value: sarama.ByteEncoder(bytesMessage),
	}
	partition, offset, err := smc.kafkaProducer.SendMessage(message)
	if err != nil {
		return err
	}
	smc.Log.Info(fmt.Sprintf("Successful message sent to partition [%d] offset [%d] with topic %s", partition, offset, topic))
	return nil
}

func (smc *KafkaClient) Predict(ctx context.Context, modelName string, host string, port int32, req payload.SeldonPayload, meta map[string][]string) (payload.SeldonPayload, error) {
	// Change client.path const values to not have prefix /
	uuid := meta[payload.SeldonPUIDHeader][0]
	bytesPayload, payloadErr := req.GetBytes()
	if payloadErr != nil {
		log.Fatalf("Error obtaining bytes payload from message: %v", payloadErr)
	}
	// TODO: Extend so predict is not hardcoded string
	inputTopic := smc.getModelInputTopicName("predict", modelName)
	callErr := smc.produceMessage(ctx, inputTopic, bytesPayload, uuid)
	// Return an empty payload given that the Kafka client will not return payload after calling
	return &payload.ProtoPayload{}, callErr
}

func (smc *KafkaClient) Chain(ctx context.Context, modelName string, msg payload.SeldonPayload) (payload.SeldonPayload, error) {
	return &payload.ProtoPayload{}, nil
}

func (smc *KafkaClient) Combine(ctx context.Context, modelName string, host string, port int32, msgs []payload.SeldonPayload, meta map[string][]string) (payload.SeldonPayload, error) {
	return &payload.ProtoPayload{}, nil
}

func (smc *KafkaClient) TransformInput(ctx context.Context, modelName string, host string, port int32, msg payload.SeldonPayload, meta map[string][]string) (payload.SeldonPayload, error) {
	return &payload.ProtoPayload{}, nil
}

func (smc *KafkaClient) TransformOutput(ctx context.Context, modelName string, host string, port int32, msg payload.SeldonPayload, meta map[string][]string) (payload.SeldonPayload, error) {
	return &payload.ProtoPayload{}, nil
}

func (smc *KafkaClient) Route(ctx context.Context, modelName string, host string, port int32, msg payload.SeldonPayload, meta map[string][]string) (int, error) {
	return 0, nil
}

func (smc *KafkaClient) Feedback(ctx context.Context, modelName string, host string, port int32, msg payload.SeldonPayload, meta map[string][]string) (payload.SeldonPayload, error) {
	return &payload.ProtoPayload{}, nil
}

func (smc *KafkaClient) Status(ctx context.Context, modelName string, host string, port int32, msg payload.SeldonPayload, meta map[string][]string) (payload.SeldonPayload, error) {
	return &payload.ProtoPayload{}, nil
}

func (smc *KafkaClient) Metadata(ctx context.Context, modelName string, host string, port int32, msg payload.SeldonPayload, meta map[string][]string) (payload.SeldonPayload, error) {
	return &payload.ProtoPayload{}, nil
}
