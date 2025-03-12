# GCP Deployment Guide

## Service Selection
1. Google Kubernetes Engine (GKE)
- Recommended for scalable agent deployment
- Supports auto-scaling based on demand
- Enables microservices architecture

2. Cloud Functions
- For serverless agent execution
- Event-driven processing
- Cost-effective for sporadic usage

## Deployment Steps

1. Container Build:
```bash
docker build -t gcr.io/[PROJECT_ID]/cybersec-agents .
docker push gcr.io/[PROJECT_ID]/cybersec-agents
```

2. GKE Deployment:
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

3. Cloud Functions:
```bash
gcloud functions deploy cybersec-agent \
    --runtime python39 \
    --trigger-http \
    --entry-point main
```