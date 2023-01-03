
about:
    @echo Test

build-docker-dev:
    docker build . --target dev --tag align3d-rusty:dev
