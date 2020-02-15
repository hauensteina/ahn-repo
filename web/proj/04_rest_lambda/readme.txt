Readme for 04_rest_lambda
============================
AHN, Feb 2020

A REST API a la Miguel Grinberg, running on AWS Lambda.

Configuration is in zappa_settings.json.

$ cat zappa_settings.json

{
    "dev": {
        "app_function": "rest-lambda.app",
        "aws_region": "us-west-2",
        "profile_name": "default",
        "project_name": "p04-rest-lambda",
        "runtime": "python3.7",
        "s3_bucket": "zappa-ahaux"
    }
}

Init with
$ python -m venv venv
$ . venv/bin/activate
$ pip install -r requirements.txt
$ zappa init

Deploy with
$ zappa deploy dev

Update with
$ zappa update dev

zappa has its own revision control. You can roll back with

$ zappa rollback dev

Read the logs with

$ zappa tail dev

For our own version control, we still use ahn-repo on github.


Testing
---------
$ url=https://qt0e49zj21.execute-api.us-west-2.amazonaws.com/dev/todo/api/v1.0
# List
$ curl -i -u anybody:Pyiaboar. $url/tasks
# Get
$ curl -i -u anybody:Pyiaboar. $url/tasks/1
# Error
$ curl -i -u anybody:Pyiaboar. curl -i -u anybody:Pyiaboar. $url/tasks/100
# Create
$ curl -i -u anybody:Pyiaboar. -H "Content-Type: application/json" -X POST -d '{"title":"Read a book"}' $url/tasks
# Update
$ curl -i -u anybody:Pyiaboar. -H "Content-Type: application/json" -X PUT -d '{"done":true}' $url/tasks/2
# Delete
$ curl -i -u anybody:Pyiaboar. -X DELETE $url/tasks/3

=== The End ===
