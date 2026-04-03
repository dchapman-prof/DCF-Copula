all:
	gcc -O3 -o histo_2d histo_2d.c -lm
	gcc -O3 -o histo histo.c -lm
	gcc -O3 -o quantile quantile.c -lm
	gcc -O3 -o fit_qdistr fit_qdistr.c qdistr.c -lm 
	gcc -O3 -o plot_qdistr plot_qdistr.c qdistr.c -lm 
	gcc -O3 -o fit_distr fit_distr.c -lm
	gcc -O3 -o table_distr table_distr.c -lm
