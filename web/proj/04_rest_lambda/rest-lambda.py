#!/usr/bin/env python

# /********************************************************************
# Filename: rest-lambda.py
# Author: AHN
# Creation Date: Feb, 2020
# **********************************************************************/
#
# A REST API stub running on AWS lambda.
# From
# https://blog.miguelgrinberg.com/post/designing-a-restful-api-with-python-and-flask
#

from pdb import set_trace as BP
from flask import Flask, jsonify, abort, make_response, request, url_for
from flask_httpauth import HTTPBasicAuth

auth = HTTPBasicAuth()
app = Flask(__name__)

tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web',
        'done': False
    }
]

# Security
#=============

@auth.get_password
#----------------------------
def get_password(username):
    if username == 'anybody':
        return 'Pyiaboar.'
    return None

@auth.error_handler
#--------------------
def unauthorized():
    return make_response(jsonify({'error': 'Unauthorized access'}), 403)

# Endpoints
#=============

@app.route('/todo/api/v1.0/tasks', methods=['GET'])
@auth.login_required
#-----------------------------------------------------
def get_tasks():
    return jsonify({'tasks': [make_public_task(task) for task in tasks]})

@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['GET'])
@auth.login_required
#-----------------------------------------------------------------
def get_task(task_id):
    task = [task for task in tasks if task['id'] == task_id]
    if len(task) == 0:
        abort(404)
    return jsonify({'task': make_public_task( task[0])})


@app.route('/todo/api/v1.0/tasks', methods=['POST'])
@auth.login_required
#-----------------------------------------------------
def create_task():
    if not request.json or not 'title' in request.json:
        abort(400)
    task = {
        'id': tasks[-1]['id'] + 1,
        'title': request.json['title'],
        'description': request.json.get('description', ""),
        'done': False
    }
    tasks.append(task)
    return jsonify({'task': make_public_task( task)}), 201

@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['PUT'])
@auth.login_required
#--------------------------------------------------------------------
def update_task(task_id):
    task = [task for task in tasks if task['id'] == task_id]
    if len(task) == 0:
        abort(404)
    if not request.json:
        abort(400)
    if 'title' in request.json and type(request.json['title']) != unicode:
        abort(400)
    if 'description' in request.json and type(request.json['description']) is not unicode:
        abort(400)
    if 'done' in request.json and type(request.json['done']) is not bool:
        abort(400)
    task[0]['title'] = request.json.get('title', task[0]['title'])
    task[0]['description'] = request.json.get('description', task[0]['description'])
    task[0]['done'] = request.json.get('done', task[0]['done'])
    return jsonify({'task': make_public_task( task[0])})

@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['DELETE'])
@auth.login_required
#---------------------------------------------------------------------
def delete_task(task_id):
    task = [task for task in tasks if task['id'] == task_id]
    if len(task) == 0:
        abort(404)
    tasks.remove(task[0])
    return jsonify({'result': True})

# Helpers
#==========

@app.errorhandler(404)
#-------------------------
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

# Replace the id field by a uri field with a URL that gets the task.
#---------------------------------------------------------------------
def make_public_task(task):
    new_task = {}
    for field in task:
        if field == 'id':
            new_task['uri'] = url_for('get_task', task_id=task['id'], _external=True)
        else:
            new_task[field] = task[field]
    return new_task

if __name__ == '__main__':
    app.run(debug=True)
