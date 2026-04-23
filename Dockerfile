FROM rayproject/ray:latest-py311-cu121

USER root

WORKDIR /app

# Copy dependency files first so the install layer is cached separately from source code.
# As long as pyproject.toml and uv.lock don't change, Docker reuses the cached layer below.
COPY pyproject.toml uv.lock ./

RUN uv pip install --system --no-cache -r pyproject.toml && \
    pip uninstall scipy -y

# Copy source code last — changes here won't invalidate the dependency install layer above.
COPY . .

USER ray

CMD ["bash"]