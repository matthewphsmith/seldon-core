package kafka

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"

	"github.com/Shopify/sarama"
	"github.com/go-logr/logr"
	"github.com/golang/protobuf/jsonpb"
	"github.com/seldonio/seldon-core/executor/api/client"
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

func NewKafkaClient(serverUrl string, protocol string, deploymentName string, predictor *v1.PredictorSpec, options ...BytesKafkaClientOption) client.SeldonApiClient {

	kafkaConfig := sarama.NewConfig()
	kafkaConfig.Producer.Retry.Max = 5
	kafkaConfig.Producer.RequiredAcks = sarama.WaitForAll
	kafkaConfig.Producer.Return.Successes = true

	producer, err := sarama.NewSyncProducer([]string{serverUrl}, kafkaConfig)
	if err != nil {
		log.Fatalf("Failed to connect to Kafka address: %v", err)
	}

	client := KafkaClient{
		producer,
		logf.Log.WithName("KafkaClient"),
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
	return fmt.Sprintf("%s-%s-%s-%s-input", smc.DeploymentName, smc.predictor.Name, modelName, method)
}

func (smc *KafkaClient) getModelInputTopicName(method string, modelName string) string {
	return fmt.Sprintf("%s-%s-%s-%s-input", smc.DeploymentName, smc.predictor.Name, modelName, method)
}

func (smc *KafkaClient) call(ctx context.Context, modelName string, method string, bytesMessage []byte, uuid string) error {
	inputTopic := smc.getModelInputTopicName(method, modelName)
	message := &sarama.ProducerMessage{
		Topic: inputTopic,
		Key:   sarama.StringEncoder(uuid),
		Value: sarama.ByteEncoder(bytesMessage),
	}
	partition, offset, err := smc.kafkaProducer.SendMessage(message)
	smc.Log.Info(fmt.Sprintf("Successful message sent to partition [%d] offset [%d] with topic %s", partition, offset, inputTopic))
	if err != nil {
		return err
	}
	return nil
}

func (smc *KafkaClient) Predict(ctx context.Context, modelName string, host string, port int32, req payload.SeldonPayload, meta map[string][]string) (payload.SeldonPayload, error) {
	// Change client.path const values to not have prefix /
	uuid := meta[payload.SeldonPUIDHeader][0]
	bytesPayload, payloadErr := req.GetBytes()
	if payloadErr != nil {
		log.Fatalf("Error obtaining bytes payload from message: %v", payloadErr)
	}
	callErr := smc.call(ctx, modelName, "predict", bytesPayload, uuid)
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
