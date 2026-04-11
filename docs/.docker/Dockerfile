FROM node:20-alpine

LABEL org.opencontainers.image.title="SKaiNET Antora" \
      org.opencontainers.image.description="Antora site generator with built-in Mermaid rendering" \
      org.opencontainers.image.source="https://github.com/SKaiNET-developers/SKaiNET-transformers"

# Chromium for mermaid-cli (puppeteer)
RUN apk add --no-cache chromium font-noto

ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser \
    PUPPETEER_SKIP_DOWNLOAD=true

WORKDIR /antora

# Install Antora + extensions + mermaid-cli in one layer
RUN npm i --save-exact \
      @antora/cli@3.1 \
      @antora/site-generator@3.1 \
      asciidoctor-kroki@0.18 \
      @mermaid-js/mermaid-cli@11 \
    && npm cache clean --force

# Mermaid-cli config: use installed Chromium, no sandbox (container)
RUN echo '{ \
  "executablePath": "/usr/bin/chromium-browser", \
  "args": ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"] \
}' > /antora/puppeteer-config.json

# Pre-generate a simple diagram to warm up and verify the stack works
RUN echo 'graph TD; A-->B;' > /tmp/test.mmd \
    && npx mmdc -i /tmp/test.mmd -o /tmp/test.svg -p /antora/puppeteer-config.json \
    && rm /tmp/test.mmd /tmp/test.svg

ENTRYPOINT ["npx", "antora"]
CMD ["--stacktrace", "antora-playbook.yml"]
