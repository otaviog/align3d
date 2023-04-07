##
# Image with all base dependencies.
FROM rust:1.66-buster as base

# For vulkano shaderc build
RUN wget https://github.com/Kitware/CMake/releases/download/v3.25.1/cmake-3.25.1-linux-x86_64.sh
RUN chmod +x cmake-3.25.1-linux-x86_64.sh && ./cmake-3.25.1-linux-x86_64.sh --skip-license

## From https://dev.to/rogertorres/first-steps-with-docker-rust-30oi
# create a new empty shell project
WORKDIR /workspaces
RUN USER=root cargo new --bin align3d
WORKDIR /workspaces/align3d

# copy over your manifests
COPY ./Cargo.lock ./Cargo.lock
COPY ./Cargo.toml ./Cargo.toml

# Cargo complains if benches is not found.
ADD ./benches ./benches
RUN touch src/lib.rs

# this build step will cache your dependencies
RUN cargo build --release
RUN cargo build

###
# Image to use with VSCode.
FROM base as dev

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -yq git emacs byobu
