# Importing
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from video_predictor import run_prediction

# Flask variables
app = Flask(__name__)
api = Api(app)

# API Reciver
class VideoPrediction(Resource):
    def post(self):
        if request.files:
            video = request.files["video"]
            res = run_prediction(video)
            return jsonify({"result" : res})
        else:
            return jsonify({"result" : "file not found"})        

# Routing
api.add_resource(VideoPrediction, '/api')

# Main 
if __name__ == '__main__':
    app.run(debug=True)