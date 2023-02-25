build:
	docker build -t 22004159g/comp5423-emotion-ui:latest .

pull:
	docker pull 22004159g/comp5423-emotion-ui:latest

run: pull
	docker run -p 8080:8080 --name 22004159G-emotion-iu -d 22004159g/comp5423-emotion-ui:latest

stop:
	docker stop 22004159G-emotion-iu && docker rm 22004159G-emotion-iu
