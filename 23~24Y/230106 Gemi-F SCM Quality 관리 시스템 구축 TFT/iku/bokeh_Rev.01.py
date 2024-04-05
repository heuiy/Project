from bokeh.io import curdoc
from bokeh.layouts import row, column, widgetbox
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput, Select
from bokeh.plotting import figure
import dataiku
import pandas as pd

# Parameterize webapp inputs
input_dataset = "Gemi"

# Set up data
mydataset = dataiku.Dataset(input_dataset)
df = mydataset.get_dataframe()

# Assume 'Batch No' and '결과값' columns are in the dataset
source = ColumnDataSource(data={
   'x': df['Batch No'],
   'y': df['결과값']
})

# Set up plot
plot = figure(plot_height=400, plot_width=400, title="My Dataiku Plot",
             tools="crosshair,pan,reset,save,wheel_zoom",
             x_axis_label='Batch No', y_axis_label='결과값')

plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

# Set up widgets
text = TextInput(title="Title", value='My Dataiku Plot')
window_size_slider = Slider(title="Window size", value=3, start=1, end=10, step=1)
dataset_select = Select(title="Dataset", value="Gemi", options=["Gemi", "OtherDataset"])

# 데이터 열 존재 여부 확인
def check_column_exists(df, column_name):
   return column_name in df.columns

# 업데이트 함수 수정
def update_data(attrname, old, new):
   window_size = window_size_slider.value

   df = mydataset.get_dataframe()

   # 'Batch No' 열이 존재하는지 확인
   if check_column_exists(df, 'Batch No') and check_column_exists(df, '결과값'):
       df.sort_values('Batch No', inplace=True)
       source.data = {
           'x': df['Batch No'],
           'y': df['결과값'].rolling(window=window_size).mean()
       }
   else:
       # 열이 없는 경우 경고 메시지 처리
       print("'Batch No' 또는 '결과값' 열이 데이터셋에 존재하지 않습니다.")

   plot.title.text = text.value

# Attach the update function to the widgets
window_size_slider.on_change('value', update_data)
text.on_change('value', update_data)
dataset_select.on_change('value', update_data)

# Set up layouts and add to document
inputs = widgetbox(text, window_size_slider, dataset_select)
curdoc().add_root(column(inputs, plot, width=800))
curdoc().title = "Dataiku Webapp"

# To run this, use 'bokeh serve --show your_script_name.py' command in the terminal

