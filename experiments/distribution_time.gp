set terminal pdfcairo size 3.4,2.4 font 'Helvetica,15'

set key outside bottom center horizontal

rate = 100000000

set xlabel 'Parameters'
set ylabel "Minimum Distribution\nTime (seconds)"
set style fill solid noborder
set xrange [1:1000]
set xtics 200
set format x '%.fM'
set logscale y

multiplier = 1000000

do for [clients in '2 25 50'] {
  set output sprintf('model-size-%d.pdf', int(clients))

  plot 2*(int(clients)*x*multiplier*8*4)/rate title 'FedAvg' w lp pt 7 ps 0.2, \
       2*(x*multiplier*0.05*8*4*(int(clients)-1))/(2*rate) title 'ERIS' w lp pt 7 ps 0.2, \
       2*(int(clients)*x*multiplier*(1-0.3)*8*4)/rate title 'PriPrune' w lp pt 7 ps 0.2, \
       2*(int(clients)*x*multiplier*0.05*8*4)/rate title 'SoteriaFL' w lp pt 7 ps 0.2
}


set xlabel 'Clients'
set ylabel "Minimum Distribution\nTime (seconds)"
set style fill solid noborder
set xrange [10:300]
set format x '%.f'
set xtics 50

do for [param in '1000000'] {
  set output sprintf('clients-%dM.pdf', int(param)/1000000)
  model_size = int(param)*8*4
  priprune_size = int(param)*(1-0.3)*8*4
  compr_size = int(param)*0.05*8*4

  plot 2*(x*model_size)/rate title 'FedAvg' w lp pt 7 ps 0.2, \
       2*(compr_size*(x-1))/(2*rate) title '2 aggr' w lp pt 7 ps 0.2, \
       2*(compr_size*(x-1))/(25*rate) title '25 aggr' w lp pt 7 ps 0.2, \
       2*(compr_size*(x-1))/(50*rate) title '50 aggr' w lp pt 7 ps 0.2, \
       2*(x*priprune_size)/rate title 'PriPrune' w lp pt 7 ps 0.2, \
       2*(x*compr_size)/rate title 'SoteriaFL' w lp pt 7 ps 0.2
}
