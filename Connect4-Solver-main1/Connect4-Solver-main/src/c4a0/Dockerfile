# === Stage 1: Build Rust/PyO3 extension ===
FROM rust:1.71 AS rust-builder
WORKDIR /app/rust

# Copy Cargo.toml trước để cache layer dependencies
COPY rust/Cargo.toml rust/Cargo.lock ./
RUN mkdir src && echo "// dummy" > src/lib.rs \
    && cargo build --release \
    && rm -rf src

# Copy code thực và build lại
COPY rust/src ./src
RUN cargo build --release

# === Stage 2: Cài Python dependencies ===
FROM python:3.11-slim AS py-builder
WORKDIR /app
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install maturin

# === Stage 3: Gói final image ===
FROM python:3.11-slim
WORKDIR /app

# Copy Python packages đã cài
COPY --from=py-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY src/requirements.txt .
COPY src/ src/

# Copy Rust extension đã biên dịch
COPY --from=rust-builder /app/rust/target/release/libc4a0.so ./c4a0.so
ENV LD_LIBRARY_PATH=/app

# Mở port và chạy
EXPOSE 8000
CMD ["python", "src/c4a0/main.py"]
