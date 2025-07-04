# Raspberry Pi WakeWord Satellite





`docker-compose.yml`
```yml
services:
  wakeword_detector:
    build:
      context: .
      args:
        - WAKEWORD_MODEL_NAME=hey_jarvis
    devices:
      - /dev/snd
    restart: unless-stopped
    ports:
      - "8700:8700" # Push Notifications
    stdin_open: true
    tty: true
```


## Permissions and Entrypoint
TODO

## Building Locally

```bash
docker build -t pipewire-docker .  
```

## Commands Explained
TODO

## Extending

```dockerfile
FROM pipewire-docker:latest

CMD ["python3", "main.py"]
```

## Credits:
* [Pipewire](https://pipewire.org)
* [ROC Toolkit](https://roc-streaming.org/toolkit/docs/)
* [Walker Griggs: PipeWire in Docker](https://walkergriggs.com/2022/12/03/pipewire_in_docker/)
