#!/bin/bash
docker build -t churn-api .
docker run -d -p 8000:8000 --name churn-api-container churn-api