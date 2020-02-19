Readme for 05_ahn_kv
============================
AHN, Feb 2020

A general purpose REST key-value store, using AWS S3, running on AWS Lambda.

Configuration is in zappa_settings.json.

$ cat zappa_settings.json

{
    "dev": {
        "app_function": "ahn-kv.app",
        "aws_region": "us-west-2",
        "profile_name": "default",
        "project_name": "ahn-kv",
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

For my own version control, I still use ahn-repo on github.

Testing
---------
$ url=https://pra3nbl6pj.execute-api.us-west-2.amazonaws.com/dev/ahn-kv
$ url=127.0.0.1:5000/ahn-kv
# Create or Update
$ curl -i -u anybody:Pyiaboar. -H "Content-Type: application/json" -X POST -d '{"key":"one", "value":1}' $url/put
# List All
$ curl -i -u anybody:Pyiaboar. $url/list_all
# List Prefix
$ curl -i -u anybody:Pyiaboar. -H "Content-Type: application/json" -X POST -d '{"prefix":"t"}' $url/list_prefix
# Get
$ curl -i -u anybody:Pyiaboar. -H "Content-Type: application/json" -X POST -d '{"key":"one"}' $url/get
# Delete
$ curl -i -u anybody:Pyiaboar. -H "Content-Type: application/json" -X POST -d '{"key":"one"}' $url/delete
# Not found
$ curl -i -u anybody:Pyiaboar. -H "Content-Type: application/json" -X POST -d '{"key":"xxx"}' $url/get

=== The End ===
