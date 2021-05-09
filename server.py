# ------------------------------------------------------
# Author : Naimish Mani B
# Date : 9th May 2021
# ------------------------------------------------------
# API Server
# https://stackoverflow.com/questions/25594893/how-to-enable-cors-in-flask
# ------------------------------------------------------

from flask import Flask, request
from flask_cors import CORS, cross_origin
import functions as f

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.before_first_request
def on_startup():
    f.init()


@app.route("/api/processRequest", methods=['GET'])
@cross_origin()
def process_request():
    args = request.get_json()
    print("args recieved", args)
    text = args['txt']
    return f.predict(text), 200


if __name__ == "__main__":
    app.run()
