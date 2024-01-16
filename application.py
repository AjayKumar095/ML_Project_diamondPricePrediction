from flask import Flask, render_template, redirect, url_for, request
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline
from logger import logging

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
  logging.info("We at home page")
  return render_template("index.html")


@app.route('/diamond_form',methods=["POST","GET"])
def diamond_form():
    if request.method=='GET':
        return render_template('diamond_form.html')
    
    else:
        data=CustomData(
            carat=float(request.form.get('carat')),
            depth = float(request.form.get('depth')),
            table = float(request.form.get('table')),
            x = float(request.form.get('x')),
            y = float(request.form.get('y')),
            z = float(request.form.get('z')),
            cut = request.form.get('cut'),
            color= request.form.get('color'),
            clarity = request.form.get('clarity')
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)
        
        results=round(pred[0],2)

        return render_template('diamond_form.html',final_result=results)
    
      
@app.route('/home')
def home():
  return redirect(url_for('index.html'))

if __name__ == '__main__':
    app.run(debug=True)