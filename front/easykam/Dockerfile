# 1단계: 빌드 단계
FROM node:22.18.0 AS build
WORKDIR /app

# package.json과 lock 파일 복사
COPY package*.json ./

# 의존성 설치
RUN npm install

# 소스 복사
COPY . .

# Vue 앱 빌드 (Vite 기본)
RUN npm run build

# 2단계: Nginx로 배포
FROM nginx:1.27.0
COPY --from=build /app/dist /usr/share/nginx/html

# Nginx 기본 설정 복사 (선택)
# COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
