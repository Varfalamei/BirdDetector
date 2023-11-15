import configparser

import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ConnectTimeoutError


class AWSUser:
    def __init__(self, access_key,
                 secret_key,
                 region='us-east-1',
                 test_bucket_name='testloaduser',
                 test_file='test.txt'
                 ):
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.test_bucket_name = test_bucket_name
        self.test_file = test_file
        self.check_connection()

    def check_connection(self):
        s3_client = boto3.client(
            service_name="s3",
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region
        )
        try:
            response = s3_client.get_object(
                Bucket=self.test_bucket_name,
                Key=self.test_file
            )
            file_content = response['Body'].read().decode('utf-8')
            if file_content != "I am health!":
                raise ValueError("Содержимое не соответствует ожидаемой строке 'I am health!'")
            else:
                print('Credentional is correct')

        except (NoCredentialsError, PartialCredentialsError, ConnectTimeoutError) as e:
            print(f"Error loading IAM client: {e}")
            return None

    def reading_file_to_s3(self, bucket_name, object_name):
        """
        Read file to s3_aws_bucket
        """
        s3_client = boto3.client('s3', aws_access_key_id=self.access_key, aws_secret_access_key=self.secret_key,
                                 region_name=self.region)

        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=object_name)
            file_content = response['Body'].read().decode('utf-8')
            print(f"Файл {object_name} успешно прочитан из  {bucket_name}")
            return file_content
        except Exception as e:
            print(f"Ошибка при загрузке файла в бакет S3: {e}")

    def upload_file_to_s3(self, bucket_name, object_name, local_file_path):
        """
        Upload a file to an S3 bucket.
        Use this function for CSV, XSL files
        """
        s3_client = boto3.client('s3', aws_access_key_id=self.access_key, aws_secret_access_key=self.secret_key,
                                 region_name=self.region)

        try:
            s3_client.upload_file(local_file_path, bucket_name, object_name)
            print(f"Файл {object_name} успешно загружен в бакет {bucket_name}")

        except Exception as e:
            print(f"Ошибка при загрузке файла в бакет S3: {e}")


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    access_key = config.get('aws-credentional', 'ACCESS-KEY')
    secret_key = config.get('aws-credentional', 'SECRET-KEY')
    aws_user = AWSUser(access_key=access_key,
                       secret_key=secret_key)

    aws_user.reading_file_to_s3('testloaduser', 'test.txt')

    aws_user.upload_file_to_s3(
        'mercedesbucket',
        'available_products.xls',
        "../dev/available_products.xls"
    )
    aws_user.upload_file_to_s3(
        'mercedesbucket',
        'the_legend_of_the_store.xls',
        "../dev/the_legend_of_the_store.xls"
    )
