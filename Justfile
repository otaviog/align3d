
about:
    @echo Test

build-docker-dev:
    docker build . --target dev --tag align3d-rusty:dev

profile_fusion:
    cargo bench --bench bench_surfel_fusion -- --profile-time=60

clippy:
    cargo clippy --verbose --all-targets --all-features -- -D warnings

fmt:
    cargo fmt --verbose --all -- --check
