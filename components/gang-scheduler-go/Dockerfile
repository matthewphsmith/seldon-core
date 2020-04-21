# Build Stage
FROM golang:1.13 as builder

WORKDIR /workspace

# Copy the Go Modules manifests
COPY go.mod go.mod
COPY go.sum go.sum

# Cache deps before building and copying source so that we don't need to
# re-download as much and so that source changes don't invalidate our
# downloaded layer
RUN go mod download

# Copy the go source
COPY . .

# Build
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 make build

# Use alpine as minimal base image to package the binary
FROM alpine:3.9.5

WORKDIR /

COPY --from=builder /workspace/bin/seldon-batch /bin
# COPY licenses/license.txt .

CMD ["sh"]
