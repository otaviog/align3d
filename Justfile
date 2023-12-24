about:
    @echo Test

bench bench_name:
    cargo bench --bench {{bench_name}}

flamegraph bench_name:
    cargo flamegraph --bench {{bench_name}}

clippy:
    cargo clippy --verbose --all-targets --all-features -- -D warnings

fmt:
    cargo  fmt --verbose --all -- --check

unit-tests:
    cargo test --verbose --release --lib