# Production image for the Next.js frontend, built in three stages so the
# final image holds only the compiled app, not the build tools.

# 1. Install dependencies. Cached until package.json or package-lock.json change.
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# 2. Build. NEXT_PUBLIC_* values are inlined into the browser JavaScript at
#    build time, so the API URL must be known here, not when the container runs.
FROM node:22-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL \
    NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# 3. Run. output: "standalone" gives a minimal server.js plus only the
#    node_modules files it needs. Static assets are copied next to it.
FROM node:22-alpine AS run
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0
COPY --from=build --chown=node:node /app/.next/standalone ./
COPY --from=build --chown=node:node /app/.next/static ./.next/static

# The node base image ships a non-root "node" user.
USER node
EXPOSE 3000
CMD ["node", "server.js"]
