FROM rayproject/ray:latest-py311-cu121

USER root

WORKDIR /app

# Copy dependency files first so the install layer is cached separately from source code.
# As long as pyproject.toml and uv.lock don't change, Docker reuses the cached layer below.
COPY pyproject.toml uv.lock ./

RUN uv pip install --system --no-cache -r pyproject.toml && \
    pip uninstall scipy -y

# no need to copy actually source files into Docker image.  When running on Ray cluster, ray.init accepts
# runtime_env.working_dir which will Ray will use to upload the source code to all worker nodes.
# This way we can avoid rebuilding Docker image every time we change source code.

# ray user is defined in base image, and has sudo privileges. We switch to ray user to avoid running the application as root.
USER ray

CMD ["bash"]