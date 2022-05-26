from flask import Flask,jsonify
from ZernikeVectors import ZernikeVectorGenerator
import json
app = Flask(__name__)


@app.route("/GenerateDataBase", methods=['GET', 'POST'])
def GiveImageGetZernikeVector():
	VectorGenerator = ZernikeVectorGenerator("C:/Users/nehdi/source/repos/TestOfConsumingRestAPI/TestOfConsumingRestAPI/Static/bardo_pcd_pics")
	data = VectorGenerator.test_GenerateZernikeVectors()
	for var in data.keys():
		keys_values = data[var].items()
		new_d = {str(key): str(value) for key, value in keys_values}
		data[var] = new_d
	json_object = json.dumps(data, indent=4)
	#with open(file_path +'/'+ file_name + ".json", "w") as outfile:
	#	outfile.write(json_object)
	return {"DataBase":json_object}





if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)


