# 前端容器：Vite 建置 → nginx 靜態服務 + WebSocket 反向代理
# 瀏覽器只面對單一來源（本容器），/ws/chat 與 /health 由 nginx 反代到 backend 容器。

# ---- 建置階段 ----
FROM node:22-alpine AS build
WORKDIR /app

# 先只複製依賴清單，讓 npm ci 能吃 build cache
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# ---- 執行階段 ----
FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
