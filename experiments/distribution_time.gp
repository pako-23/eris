set terminal pdfcairo size 3.4,2.4 font 'Helvetica,15'

set key outside bottom horizontal

rate = 100000000

set xlabel 'Parameters'
set ylabel "Minimum Distribution\nTime (seconds)"
set style fill solid noborder
set xrange [1:1000]
set xtics 200
set samples 30
set format x '%.fM'
set logscale y

multiplier = 1000000

do for [clients in '50'] {
  set output sprintf('model-size-%d.pdf', int(clients))

  plot 2*(int(clients)*x*multiplier*8*4)/rate title 'FedAvg' w lp pt 7 ps 0.3 lc '#cc52cc', \
       2*(x*multiplier*0.05*8*4*(int(clients)-1))/(2*rate) title 'ERIS (A = 2)' w lp pt 7 ps 0.3 lc '#3030aa', \
       2*(x*multiplier*0.05*8*4*(int(clients)-1))/(25*rate) title 'ERIS (A = 25)' w lp pt 7 ps 0.3 lc '#5252cc', \
       2*(x*multiplier*0.05*8*4*(int(clients)-1))/(50*rate) title 'ERIS (A = 50)' w lp pt 7 ps 0.3 lc '#7474ee', \
       2*(int(clients)*x*multiplier*(1-0.3)*8*4)/rate title 'PriPrune' w lp pt 7 ps 0.3 lc '#52cc52', \
       2*(int(clients)*x*multiplier*0.05*8*4)/rate title 'SoteriaFL' w lp pt 7 ps 0.3 lc '#cc5252', \
       (x*multiplier*8*4)/rate title 'Ako' w lp pt 7 ps 0.3 lc '#cccc52'
}


set xlabel 'Clients'
set ylabel "Minimum Distribution\nTime (seconds)"
set style fill solid noborder
set xrange [10:300]
set format x '%.f'
set xtics 50

do for [param in '10000000'] {
  set output sprintf('clients-%dM.pdf', int(param)/1000000)
  model_size = int(param)*8*4
  priprune_size = int(param)*(1-0.3)*8*4
  compr_size = int(param)*0.05*8*4

  plot 2*(x*model_size)/rate title 'FedAvg' w lp pt 7 ps 0.3 lc '#cc52cc', \
       2*(compr_size*(x-1))/(2*rate) title 'ERIS (A = 2)' w lp pt 7 ps 0.3 lc '#3030aa', \
       2*(compr_size*(x-1))/(25*rate) title 'ERIS (A = 25)' w lp pt 7 ps 0.3 lc '#5252cc', \
       2*(compr_size*(x-1))/(50*rate) title 'ERIS (A = 50)' w lp pt 7 ps 0.3 lc '#7474ee', \
       2*(x*priprune_size)/rate title 'PriPrune' w lp pt 7 ps 0.3 lc '#52cc52', \
       2*(x*compr_size)/rate title 'SoteriaFL' w lp pt 7 ps 0.3 lc '#cc5252', \
       model_size/rate title 'Ako' w lp pt 7 ps 0.3 lc '#cccc52'
}
