## Инструкция по деплою FAST-API сервиса на EC2 

###  Cоздаем образ инстанса на AWS
- AWS main -> Instances -> Launch instances ->  Create
- Выбираем образ Ubuntu
- В firewall установить securite politics ( При тестовым режиме оставить обычную и добавить два allow)
- Создаем скачиваем файлик key-pair 

### Подключаемся через SSH к инстансу 
- для windows icacls fastapi_key.pem /inheritance:r /grant:r "%USERNAME%:R"
- ssh -i "fastapi_key.pem" <Адрес нашего инстанста>.amazonaws.com


### Внутри instance EC2

```
sudo apt-get update
sudo apt install -y python3-pip nginx
sudo vim /etc/nginx/sites-enabled/fastapi_nginx

server {
    listen 80;   
    server_name <YOUR_EC2_IP>;    
    location / {        
        proxy_pass http://127.0.0.1:8000;    
    }
}
sudo service nginx restart

git clone -b dev_mers https://github.com/Varfalamei/BirdDetector.git 
cd BirdDetector/example_docker_fastapi_for_ec2
pip3 install uvicorn fastapi
python3 -m uvicorn main:app
```

