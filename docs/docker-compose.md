# Docker Compose file

This Docker Compose file defines two services:

## localgpt service

This service builds the localgpt image and runs the localGPT API server.

```yaml
services:
  localgpt:
    build: 
      context: .
      dockerfile: Dockerfile.api
    image: localgpt
    environment: 
      - device_type=${DEVICE_TYPE}
      - source_documents=${SOURCE_DOCUMENTS}
    volumes:
      - "$HOME/.cache:/root/.cache"
      - "$HOME/${SOURCE_DOCUMENTS}:/SOURCE_DOCUMENTS"  
    command: python run_localGPT_API.py --port 8080 --host 0.0.0.0
    networks:
      - localgpt-network
```

- Builds Dockerfile.api 
- Sets device_type and source_documents environment variables
- Mounts host cache and source documents
- Runs the API server on port 8080

## localgpt-ui service

This service runs the localGPT UI frontend.

```yaml
  localgpt-ui:
    build: 
      context: localGPTUI/  
      dockerfile: Dockerfile
    ports:
      - "5111:5111" 
    environment:
      API_HOST: http://localgpt:8080/api
    depends_on:
      - localgpt
    networks:
      - localgpt-network
```

- Builds the localGPTUI Dockerfile
- Exposes port 5111 
- Sets API_HOST env var pointing to localgpt API
- Depends on localgpt service

## Network

```yaml
networks:
  localgpt-network:
    driver: bridge
```

- Single network for communication between containers
