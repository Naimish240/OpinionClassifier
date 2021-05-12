# ------------------------------------------------------
# Author : Naimish Mani B
# Date : 9th May 2021
# ------------------------------------------------------
# API Server
# https://stackoverflow.com/questions/25594893/how-to-enable-cors-in-flask
# ------------------------------------------------------

from flask import Flask, request
from flask_cors import CORS, cross_origin
from . import functions as f


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
f.init()

'''
@app.before_first_request
def on_startup():
    f.init()
'''


@app.before_request()
def check():
    if request.method == 'OPTIONS':
        return Flask.Response(status=200)


@app.route("/")
def home():
    return "Server is running"


@app.route("/api/processRequest", methods=['POST'])
def process_request():
    text = request.args.get('text')
    print("text recieved", text)
    return f.predict(text), 200


if __name__ == "__main__":
    app.run()
