package kafka

import (
	"io"
	"net/http"

	"github.com/Shopify/sarama"
	"github.com/go-logr/logr"
	"github.com/golang/protobuf/jsonpb"
	"github.com/seldonio/seldon-core/executor/api/grpc/seldon/proto"
	"github.com/seldonio/seldon-core/executor/api/metric"
	"github.com/seldonio/seldon-core/executor/api/payload"
	v1 "github.com/seldonio/seldon-core/operator/apis/machinelearning/v1"
)

const (
	ContentTypeJSON = "application/json"
)

type KafkaClient struct {
	kafkaClient    sarama.AsyncProducer
	Log            logr.Logger
	Protocol	   string
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

func BytesKafkaClientOption func(client *KafkaClient)

func NewKafkaClient(hostname string, port int, protocol string, deployName string, predictor *v1.PredictorSpec, options ...BytesKafkaClientOption) client.SeldonApiClient {

	kafkaConfig := sarama.NewConfig()
	config.Producer.Retry.Max = 5
	config.Producer.RequiredAcks = sarama.WaitForAllAll
	config.Producer.Return.Successes = true

	producer, err := sarama.NewAsyncProducer([]string{fmt.Sprintf("%s:%d", hostname, port), config)
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

func (smc *KafkaClient) getInputTopicName(method string, modelName string) string {
	return fmt.Sprintf("%s-%s-%s-%s-input", smc.DeploymentName, smc.Predictor.Name, modelName, method)
}

func (smc *KafkaClient) call(ctx context.Context, modelName string, method string, msg []byte, uuid string) error {
	inputTopic := smc.getInputTopicName(method, modelName)
	msg := &smc.kafkaClient.ProducerMessage {
		Topic: inputTopic,
		Key: uuid,
		Value: msg,
	}
	partition, offset, err := smc.kafkaClient.SendMessage(msg)
	if err != nil {
		return err
	}
}

func (smc *KafkaClient) Predict(ctx context.Context, modelName string, host string, port int32, req payload.SeldonPayload, meta map[string][]string) (payload.SeldonPayload, error) {
	// Change client.path const values to not have prefix / 
	return payload.SeldonPayload{}, smc.call(ctx, modelName, "predict", req.GetPayload(), meta[payload.SeldonPUIDHeader])
}
