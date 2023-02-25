FROM node:18.14.2-alpine3.17 AS BUILD

WORKDIR /libs

COPY package.*  package.*

RUN yarn

FROM node:18.14.2-alpine3.17 AS APP

WORKDIR /app

COPY --from=BUILD /libs/* /app/*

COPY . .

ENTRYPOINT [ "yarn", "serve" ]