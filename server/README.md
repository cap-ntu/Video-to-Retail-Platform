# Hysia-V2O RESTful Server

This module contains the implementation of Hysia-V2O's backend processing logic. The database design 
and request handling is supported by Django Rest Framework.

![Alt text](static/media/server-system-architecture.jpg?raw=true "server system architecture")

## Features

- **RESTful APIs**: a set of APIs to achieve management, video processing and scene search.
- **RPC**: using gRPC protocol to communicate with model servers.
- **Multiple model servers**: a set of servers which handles ML requests sent from Django.

## Installation

Initially, we need to generate python-runnable gRPC code reset Django database by:
```shell script
python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. protos/api2msl.proto
```
And reset database:
```shell script
bash reset-db.sh
```

## Demo

Start model servers by:
```shell script
python start_model_servers
```

And start Django by:
```shell script
python manage.py runserver
```

### RESTful APIs

The `restapi` folder contains the implementation of RESTful APIs (`restapi/views.py`), video processing logic (`restapi/processing.py`) 
and video searching logic (`restapi/search.py`)

### Model Servers

The `model_server` folder contains different RPC servers each hosts a deep learning model, which handles ML
requests from Django. The communication between Django and model servers follows Google Remote Procedure Calls (gRPC) protocol, defined as:
```proto
syntax = "proto3";

service Api2Msl {
  /* Interface for sending encoded video frame and get predicted Json path */
  rpc GetJson(BundleRequest) returns (JsonReply) {}
}

message BundleRequest {
  /* Bytes (etc. encoded frame) */
  bytes buf = 1;
  /* Meta data */
  string meta = 2;
}

message JsonReply {
  /* Json as string */
  string json = 1;
  /* Meta data */
  string meta = 2;
}
```
