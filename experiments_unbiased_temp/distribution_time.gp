set terminal pdfcairo size 3.4,2.4 font 'Helvetica,15'

rate = 100000000

set key top left
set xlabel 'Clients'
set ylabel "Minimum Distribution\nTime (seconds)"
set style fill solid noborder
set xrange [20:100]


do for [param in '1000000 10000000 100 000 000'] {
  set output sprintf('model-size-%dM.pdf', int(param)/1000000)
  model_size = int(param)*8*4

  plot 2*(x*model_size)/rate title 'Centralized FL' w lp pt 7 ps 0.2, \
       2*(model_size*(x-1))/(2*rate) title '2 aggregators' w lp pt 7 ps 0.2, \
       2*(model_size*(x-1))/(5*rate) title '5 aggregators' w lp pt 7 ps 0.2, \
       2*(model_size*(x-1))/(10*rate) title '10 aggregators' w lp pt 7 ps 0.2
}
