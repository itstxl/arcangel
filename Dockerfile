# ArcAngel in a container: nothing but the GPU kernel driver (xe, Linux 6.12+) is needed on the host.
#
#   docker build -t arcangel .
#   docker run -d --name arcangel --device /dev/dri -p 8082:8082 \
#     -v /path/to/models:/models arcangel --model /models/Inkling-2-Mini-UD-Q4_K_XL.gguf --n-ctx 262144 --kv-bits 8
#
# Any inkling_server.py option can follow the image name. The offload recipe also needs
# `-e INK_OFFLOAD=0-5,18-23 --ulimit memlock=-1` (pinned system RAM for the experts).
# The build stage holds the oneAPI compiler (~8 GB); the image you run only gets its runtime libraries.

ARG UBUNTU=24.04

# ---- build: oneAPI DPC++ + ocloc, kernels compiled ahead of time for every Battlemage chip --------------
FROM ubuntu:${UBUNTU} AS build
ARG ONEAPI=2026.1
ARG AOT=bmg_g21,bmg_g31
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl gpg cmake ninja-build g++ software-properties-common \
 && curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
      | gpg --dearmor -o /usr/share/keyrings/oneapi.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/oneapi.gpg] https://apt.repos.intel.com/oneapi all main" \
      > /etc/apt/sources.list.d/oneapi.list \
 && add-apt-repository -y ppa:kobuk-team/intel-graphics \
 && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      intel-oneapi-compiler-dpcpp-cpp-${ONEAPI} intel-ocloc \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt ./
COPY src src
COPY tools tools
COPY tests tests
RUN . /opt/intel/oneapi/setvars.sh >/dev/null \
 && cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release -DINK_AOT_TARGETS=${AOT} \
 && cmake --build build --target inkling ink-probe bench-decode bench-prefill \
 && mkdir -p /out/lib /out/build \
 && cp build/libinkling.so build/ink-probe build/bench-decode build/bench-prefill /out/build/ \
 && cd /opt/intel/oneapi/compiler/latest/lib \
 && cp -P libsycl.so* libur_loader.so* libur_adapter_level_zero*.so* libsvml.so libimf.so libirng.so \
      libintlc.so* /out/lib/ \
 && cp -P /opt/intel/oneapi/umf/latest/lib/libumf.so* /opt/intel/oneapi/tcm/latest/lib/libhwloc.so* /out/lib/

# ---- run: Level Zero GPU runtime, the engine, the Python server ---------------------------------------
FROM ubuntu:${UBUNTU}
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl gpg software-properties-common python3 python3-venv \
 && add-apt-repository -y ppa:kobuk-team/intel-graphics \
 && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libze1 libze-intel-gpu1 \
 && apt-get purge -y software-properties-common && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
# the server's requirements.txt minus huggingface_hub (only install.sh's downloader uses it)
RUN mkdir -p /app/server && python3 -m venv /app/venv && /app/venv/bin/pip install --no-cache-dir -q jinja2 numpy tokenizers
# tokenizer and chat template of thinkingmachines/Inkling-2-Mini, pinned to the versions install.sh checks
RUN cd /app/server \
 && curl -fsSL --retry 3 -O https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/tokenizer.json \
 && curl -fsSL --retry 3 -O https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/chat_template.jinja \
 && printf '%s  %s\n' \
      7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6 tokenizer.json \
      53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc chat_template.jinja | sha256sum -c
COPY --from=build /out/lib /app/lib
COPY --from=build /out/build /app/build
COPY server/inkling_server.py /app/server/
ENV LD_LIBRARY_PATH=/app/lib PATH=/app/build:$PATH ONEAPI_DEVICE_SELECTOR=level_zero:*
EXPOSE 8082
ENTRYPOINT ["/app/venv/bin/python", "-u", "/app/server/inkling_server.py", "--host", "0.0.0.0"]
