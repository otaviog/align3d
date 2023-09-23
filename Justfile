about:
    @echo Test

profile_fusion:
    cargo bench --bench bench_surfel_fusion -- --profile-time=60

clippy:
    cargo clippy --verbose --all-targets --all-features -- -D warnings

fmt:
    cargo  fmt --verbose --all -- --check

unit-tests:
    cargo test --verbose --release --lib