from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
import calendar
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///events.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Event(db.Model):
   id = db.Column(db.Integer, primary_key=True)
   date = db.Column(db.String(50), nullable=False)
   manufacture_number = db.Column(db.String(100))
   time = db.Column(db.String(50))
   production_manager = db.Column(db.String(100))
   qc_test_director = db.Column(db.String(100))

   def to_dict(self):
       return {
           'date': self.date,
           'manufacture_number': self.manufacture_number,
           'time': self.time,
           'production_manager': self.production_manager,
           'qc_test_director': self.qc_test_director
       }

@app.route('/', methods=['GET', 'POST'])
def index():
   if request.method == 'POST':
       # 여기서 Event 인스턴스 생성 로직을 구현하세요.
       pass

   year = 2024
   month = 1
   cal_data = calendar.Calendar().monthdayscalendar(year, month)
   events = {event.date: event.to_dict() for event in Event.query.all()}
   return render_template('index.html', year=year, month=month, cal_data=cal_data, events=events)

@app.route('/add_event', methods=['POST'])
def add_event():
   # 폼 데이터 가져오기
   date = request.form['date']
   manufacture_number = request.form['manufacture_number']
   time = request.form['time']
   production_manager = request.form['production_manager']
   qc_test_director = request.form['qc_test_director']

   # 새 이벤트 생성 및 저장
   new_event = Event(
       date=date,
       manufacture_number=manufacture_number,
       time=time,
       production_manager=production_manager,
       qc_test_director=qc_test_director
   )
   db.session.add(new_event)
   db.session.commit()

   return redirect('/')

if __name__ == '__main__':
   with app.app_context():
       db.create_all()
   app.run(debug=True)
