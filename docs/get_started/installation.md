# 🚀 Get Started

Choose one of the following methods to set up your environment. We highly recommend using Docker for development or production environment.

## 📦 Option 1: Installation

Please make sure you have built and installed [`ucx`](https://github.com/openucx/ucx) in your environment.

```bash
# clone this repository
git clone git@github.com:sgl-project/sglang-omni.git
cd sglang-omni

# create a virtual environment
uv venv .venv -p 3.12
source .venv/bin/activate

# install
uv pip install -v .

# install for development
uv pip install -v -e .
```

## 🐳 Option 2: Use Docker

We have build all necessary dependencies into our Docker Image, so you can simply pull and run it.

```bash
# we strongly recommend using our docker image for stable environment
# NOTE: this docker image will be moved to lmsysorg upon release
docker pull frankleeeee/sglang-omni:dev

# run the container
docker run -it \
    --shm-size 32g \
    --gpus all \
    --ipc host \
    --network host \
    --privileged \
    frankleeeee/sglang-omni:dev \
    /bin/zsh
```
