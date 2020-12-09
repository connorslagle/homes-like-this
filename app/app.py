from flask import Flask, render_template, Response, redirect, url_for, request, flash
import time
from datetime import datetime
import os
import atexit
import pickle
import requests
import pandas as pd
import numpy as np

from apscheduler.schedulers.background import BackgroundScheduler
from flask_pymongo import PyMongo

# for uploading image
from werkzeug.utils import secure_filename

'''
For uploading image
'''
UPLOAD_FOLDER = './static/images/fromuser'
ALLOWED_EXTENSIONS = {'jpeg','png'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


'''
Start App
'''
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config["MONGO_URI"] = "mongodb://localhost:27017/"


'''
Pages
'''
@app.route('/')
def index():
    return render_template('cover.html')

@app.route('/submit', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return render_template('submit.html')
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return render_template('submit.html')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # TODO: load model
            # open model
            # with open('models/test_model.pkl', 'rb') as f:
            #     model = pickle.load(f)

            # TODO: load data, predict on data, return top 10 imgs and locations
            # pass to results page
            

            return render_template('results.html', filename=filename)
    else:
        return render_template('submit.html')

# TODO results page
# @app.route('/results')

# TODO aboutme page w/LI and GH links
# @app.route('/aboutme')
# def get_and_score():
#     new_json = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point')
#     stream_json = new_json.json()

#     return render_template('table.html',listing=test.json())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8105, threaded=True, debug=True)
    

    '''
    Trying to stream data
    '''
    # def tick():
    #     return '\ntesting stream'

    # scheduler = BackgroundScheduler()
    # scheduler.add_job(func=tick, trigger='interval',seconds=5)
    # scheduler.start()
    # atexit.register(lambda: scheduler.shutdown())

    # # from https://flask.palletsprojects.com/en/1.1.x/patterns/streaming/#streaming-from-templates
    # def stream_template(template_name, **context):
    #     app.update_template_context(context)
    #     t = app.jinja_env.get_template(template_name)
    #     rv = t.stream(context)
    #     rv.enable_buffering(5)
    #     return rv

    # get incoming data from API
    # @app.route('/stream')
    # def test():
    #     return Response(stream_template('table.html',listing=test))



    # # from https://flask.palletsprojects.com/en/1.1.x/patterns/streaming/#streaming-from-templates
    # def stream_template(template_name, **context):
    #     app.update_template_context(context)
    #     t = app.jinja_env.get_template(template_name)
    #     rv = t.stream(context)
    #     rv.enable_buffering(5)
    #     return rv
