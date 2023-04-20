<script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>

<script type="text/javascript">
  google.charts.load("current", {packages:["timeline"]});
  google.charts.setOnLoadCallback(drawChart);
  function drawChart() {

    var container = document.getElementById('example3.1');
    var chart = new google.visualization.Timeline(container);
    var dataTable = new google.visualization.DataTable();
    dataTable.addColumn({ type: 'string', id: 'Position' });
    dataTable.addColumn({ type: 'string', id: 'Name' });
    dataTable.addColumn({ type: 'date', id: 'Start' });
    dataTable.addColumn({ type: 'date', id: 'End' });
    dataTable.addRows([
      [ 'DPX23001', '반응기_01', new Date(1789, 3, 30), new Date(1797, 2, 4) ],
      [ 'DPX23001', '반응기_02', new Date(1797, 2, 4), new Date(1801, 2, 4) ],
      [ 'DPX23001', 'Thomas Jefferson', new Date(1801, 2, 4), new Date(1809, 2, 4) ],
      [ 'DPX23002', 'John Adams', new Date(1789, 3, 21), new Date(1797, 2, 4)],
      [ 'DPX23002', 'Thomas Jefferson', new Date(1797, 2, 4), new Date(1801, 2, 4)],
      [ 'DPX23002', 'Aaron Burr', new Date(1801, 2, 4), new Date(1805, 2, 4)],
      [ 'DPX23002', 'George Clinton', new Date(1805, 2, 4), new Date(1812, 3, 20)],
      [ 'DPX23003', 'John Jay', new Date(1789, 8, 25), new Date(1790, 2, 22)],
      [ 'DPX23003', 'Thomas Jefferson', new Date(1790, 2, 22), new Date(1793, 11, 31)],
      [ 'DPX23003', 'Edmund Randolph', new Date(1794, 0, 2), new Date(1795, 7, 20)],
      [ 'DPX23003', 'Timothy Pickering', new Date(1795, 7, 20), new Date(1800, 4, 12)],
      [ 'DPX23003'', 'Charles Lee', new Date(1800, 4, 13), new Date(1800, 5, 5)],
      [ 'DPX23003', 'John Marshall', new Date(1800, 5, 13), new Date(1801, 2, 4)],
      [ 'DPX23003', 'Levi Lincoln', new Date(1801, 2, 5), new Date(1801, 4, 1)],
      [ 'DPX23003', 'James Madison', new Date(1801, 4, 2), new Date(1809, 2, 3),
      [ 'DPX23004', 'Levi Lincoln', new Date(1801, 2, 5), new Date(1801, 4, 1)],
      [ 'DPX23004', 'Levi Lincoln', new Date(1801, 2, 5), new Date(1801, 4, 1)],
      [ 'DPX23005', 'Levi Lincoln', new Date(1801, 2, 5), new Date(1801, 4, 1)],
      [ 'DPX23005', 'Levi Lincoln', new Date(1801, 2, 5), new Date(1801, 4, 1)],
      [ 'DPX23006', 'Levi Lincoln', new Date(1801, 2, 5), new Date(1801, 4, 1)],
      [ 'DPX23006', 'Levi Lincoln', new Date(1801, 2, 5), new Date(1801, 4, 1)]]
    ]);

    chart.draw(dataTable);
  }
</script>

<div id="example3.1" style="height: 200px;"></div>