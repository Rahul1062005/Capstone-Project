# Multi-stage Java 21 Spring Boot build for Render / Docker
FROM eclipse-temurin:21-jdk-alpine AS builder
WORKDIR /app

COPY gradlew settings.gradle build.gradle gradle.properties ./
COPY gradle ./gradle
RUN chmod +x gradlew

COPY src ./src
RUN ./gradlew bootJar --no-daemon -x test

FROM eclipse-temurin:21-jre-alpine
WORKDIR /app

COPY --from=builder /app/build/libs/*-SNAPSHOT.jar /app/app.jar

ENV PORT=5000
EXPOSE 5000

ENTRYPOINT ["sh", "-c", "java -XX:+UseSerialGC -Xmx256m -Xss512k -Dserver.port=${PORT:-5000} -jar /app/app.jar"]
