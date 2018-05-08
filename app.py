from flask import Flask, request, abort, jsonify, Response, send_file
import cv2
import jsonpickle
import numpy as np
import test_frcnn_score as ts
import midi_generate as md

app = Flask(__name__)


#app.config["DEBUG"] = True

@app.route("/")
def home():
	return "music score flask api!"


@app.route('/api/get_score', methods=['POST'])
def check_fashion_status():
	nparr = np.fromstring(request.data, np.uint8)
	img = cv2.imdecode(nparr, 1)
	cv2.imwrite('./server_files/received.jpeg', img)
	boxes, class_mapping = ts.predict_from_server("./server_files/received.jpeg")
	# print (boxes, class_mapping)
	md.run_midi_generate(boxes, class_mapping)
	response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])}
	# encode response using jsonpickle
	response_pickled = jsonpickle.encode(response)
	myMidi = open('./server_files/output.mid', 'rb')
	return send_file(myMidi, mimetype="audio/midi")
	# return Response(response=response_pickled, status=200, mimetype="application/json")

if __name__ == "__main__":
	app.run(debug=True)

