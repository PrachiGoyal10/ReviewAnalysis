from flask import Flask, render_template,session,redirect,url_for
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from ReviewAnalysis_API import *
#from flask_session import Session


app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkeyisgreat'

#sess = Session()


class myform(FlaskForm) :
    name = StringField(" Enter the complete folder location: ", validators=[DataRequired()])
    submit = SubmitField()

class demoform(FlaskForm):
    name= StringField("Test me ?")
    submitme = StringField()


@app.route("/",methods=['GET','POST'])
def webpage():
    Form = myform()
    if Form.validate_on_submit():
        session['folder_name'] = Form.name.data
        return redirect(url_for('Features'))
    return render_template('main.html',form=Form)



@app.route("/Features",methods=['GET','POST'])
def Features():
    name = session['folder_name']
    neg,pos,ttlrating, ttlreviews,vrsnrating,latestvrsn,img = getFeature(name)
    return render_template('feature.html' ,best_negsentences = neg,pos_best_sentences = pos,total_rating=ttlrating,total_reviews=ttlreviews,latest_version=latestvrsn,VrsnRating=vrsnrating,image=img)



if __name__ ==  "__main__":
    try:
        app.run(host="0.0.0.0",port="3050",debug=True)
    except Exception as e:
        print(" Exception occurred at Main "+str(e))