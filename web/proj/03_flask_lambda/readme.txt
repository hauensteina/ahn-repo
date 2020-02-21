Readme for 03_flask_lambda
============================
AHN, Feb 2020

An endpoint returning 'Hello World' in Json, deployed to
AWS LAmbda and active at

https://zzwt4p3ski.execute-api.us-west-2.amazonaws.com/dev

Configuration is in zappa_settings.json.

$ cat zappa_settings.json

{
    "dev": {
        "app_function": "flask-lambda.app",
        "aws_region": "us-west-2",
        "profile_name": "default",
        "project_name": "p03-flask-lambda",
        "runtime": "python3.7",
        "s3_bucket": "zappa-ahaux"
    }
}


Deploy with
$ zappa deploy dev

Update with
$ zappa update dev

zappa has its own revision control. You can roll back with

$ zappa rollback dev

Read the logs with

$ zappa tail dev

=== The End ===
