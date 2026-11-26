# syntax=docker/dockerfile:1
FROM ubuntu:24.04 AS runtime-base

ARG TARGETARCH
# The 2026.2.0-863 compiler runtime package omits the Level Zero adapters.
ARG ONEAPI_VERSION=2026.1.1-325

RUN test "$TARGETARCH" = amd64 || { echo 'ArcAngel requires linux/amd64' >&2; exit 1; }
# Keep downloaded .debs in BuildKit's cache instead of letting the base image purge them.
RUN --mount=type=cache,id=arcangel-noble-apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=arcangel-noble-apt-lists,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB -o /etc/apt/keyrings/intel-oneapi.asc \
    && curl -fsSL 'https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x0C0E6AF955CE463C03FC51574D098D70AFBE5E1F' -o /etc/apt/keyrings/intel-graphics.asc \
    && echo 'deb [signed-by=/etc/apt/keyrings/intel-oneapi.asc] https://apt.repos.intel.com/oneapi all main' > /etc/apt/sources.list.d/intel-oneapi.list \
    && echo 'deb [signed-by=/etc/apt/keyrings/intel-graphics.asc,0C0E6AF955CE463C03FC51574D098D70AFBE5E1F] https://ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu noble main' > /etc/apt/sources.list.d/intel-graphics.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        "intel-oneapi-compiler-dpcpp-cpp-runtime-2026.1=$ONEAPI_VERSION" libhwloc15 libze-intel-gpu1 libze1 python3

COPY <<EOF /etc/ld.so.conf.d/oneapi.conf
/opt/intel/oneapi/compiler/latest/lib
/opt/intel/oneapi/tbb/latest/lib
/opt/intel/oneapi/umf/latest/lib
EOF
RUN ldconfig

FROM runtime-base AS build

RUN --mount=type=cache,id=arcangel-noble-apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=arcangel-noble-apt-lists,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential cmake "intel-oneapi-compiler-dpcpp-cpp-2026.1=$ONEAPI_VERSION" intel-ocloc ninja-build python3-venv

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=server/requirements.txt,target=/tmp/requirements.txt \
    python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --disable-pip-version-check --only-binary=:all: -r /tmp/requirements.txt

COPY server/chat_template.sha256 server/tokenizer.sha256 /assets/
WORKDIR /assets
# Both assets come from the same Hugging Face revision. The GGUF is mounted at runtime.
RUN base=https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/c9d028302cd0f035fdf8e7e91f089a4ce94d2195 \
    && curl -fsSL --retry 3 "$base/tokenizer.json" -o tokenizer.json \
    && curl -fsSL --retry 3 "$base/chat_template.jinja" -o chat_template.jinja \
    && printf '%s  %s\n' "$(cat tokenizer.sha256)" tokenizer.json "$(cat chat_template.sha256)" chat_template.jinja | sha256sum -c

WORKDIR /src

# bmg_g31 is required for Arc Pro B70; the other profiles use bmg_g21.
ARG INK_AOT_TARGETS=bmg_g21
RUN --mount=type=bind,source=CMakeLists.txt,target=/src/CMakeLists.txt \
    --mount=type=bind,source=src,target=/src/src \
    --mount=type=bind,source=tools,target=/src/tools \
    --mount=type=bind,source=tests,target=/src/tests \
    cmake -S . -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icpx \
        -DINK_AOT_TARGETS="$INK_AOT_TARGETS" \
    && cmake --build /build --target inkling

FROM build AS diagnostics-build

RUN --mount=type=bind,source=CMakeLists.txt,target=/src/CMakeLists.txt \
    --mount=type=bind,source=src,target=/src/src \
    --mount=type=bind,source=tools,target=/src/tools \
    --mount=type=bind,source=tests,target=/src/tests \
    cmake --build /build --target ink-probe bench-decode bench-prefill

FROM runtime-base AS serving

WORKDIR /opt/arcangel
COPY --link --from=build /opt/venv /opt/venv
COPY --link --from=build /build/libinkling.so build/libinkling.so
COPY --link --from=build /assets/tokenizer.json /assets/chat_template.jinja /assets/chat_template.sha256 server/
COPY server/inkling_server.py server/

ENV PYTHONDONTWRITEBYTECODE=1
USER 10001:10001

# SYCL loads the Level Zero adapter on demand, so loading the engine alone won't catch a missing adapter.
RUN ["/opt/venv/bin/python", "-c", "import ctypes; ctypes.CDLL('build/libinkling.so'); ctypes.CDLL('libur_adapter_level_zero_v2.so.0'); ctypes.CDLL('libze_loader.so.1'); ctypes.CDLL('libze_intel_gpu.so.1')"]

EXPOSE 8082
ENTRYPOINT ["/opt/venv/bin/python", "-u", "server/inkling_server.py", "--host", "0.0.0.0"]
CMD ["--help"]

FROM serving AS diagnostics

COPY --link --from=diagnostics-build /build/ink-probe /build/bench-decode /build/bench-prefill build/

FROM serving AS runtime
