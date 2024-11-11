set terminal pdfcairo font 'Helvetica,15'

rate = 500000000

set key top left
set xlabel 'Clients'
set ylabel "Minimum Model\nDistribution Time"
set style fill solid noborder
set xrange [1:1000]


do for [param in '1000000 10000000 100000000'] {
  set output sprintf('model-size-%dM.pdf', int(param)/1000000)
  model_size = int(param)*8*4

  plot 2*(x*model_size)/rate title 'FedAvg' w lp pt 7 ps 0.2, \
       2*(model_size*x)/(5*rate) title 'eris 5' w lp pt 7 ps 0.2, \
       2*(model_size*x)/(10*rate) title 'eris 10' w lp pt 7 ps 0.2, \
       2*(model_size*x)/(20*rate) title 'eris 20' w lp pt 7 ps 0.2
}
