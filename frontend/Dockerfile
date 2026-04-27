FROM oven/bun:1 AS deps
WORKDIR /app
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

FROM deps AS build
WORKDIR /app
COPY . .
RUN bun run build

FROM oven/bun:1 AS runner
WORKDIR /app
COPY --from=build /app /app
EXPOSE 5173
CMD ["bun", "run", "preview", "--host", "0.0.0.0", "--port", "5173"]
