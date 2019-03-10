#include <stdio.h>
#include <omp.h>
#include <math.h>
#define N 400
#define M 400
#define tolerance (1e-7)

double S(double x, double y){
	return -10*(x*x+y*y+5);
}


void print(double u[N][M]){
	int i,j;
	for(i=0; i<N; i++){
		for(j=0; j<M; j++){
			printf("%.2lf\t",u[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

int main(){
        int nThreads;
	double u[N][M];
	double xmin = 0.0;
	double xmax = 1.0;
	double ymin = 0.0;
	double ymax = 1.0;
	double h = (xmax-xmin)/(N-1);
	double k = (ymax-ymin)/(M-1);
	int iterations = 0;
	
	int i = 0;
	int j = 0;
	
	double uNew[N][M];
	double error = 0.1;
	double sum = 0.0;
	
	for(i=0; i<N; i++){
		for(j=0; j<M; j++){
			if(j==M-1) {
				u[i][j] = 1;
				uNew[i][j] = 1;
			}
			else {
				u[i][j] = 0;
				uNew[i][j] = 0;
			}
		}
	}

	
//	print(u);
	
	#pragma omp parallel firstprivate(nThreads) num_threads(4)
        {
            nThreads = omp_get_num_threads();
            if (omp_get_thread_num()==0)
            printf("# of threads: %d\n",nThreads);
        }
	
//	FILE *ferror = fopen("LiebmannError.txt","w");
	
	// Liebmann
	printf("error = %lf , tol = %.7lf\n",error,tolerance);
        double start_time = omp_get_wtime();
	while(error>tolerance){
		sum = 0.0;
                #pragma omp parallel for shared(uNew,u) private(i,j) firstprivate(h,k) reduction(+:sum) num_threads(4)		
		for(i=1; i<N-1; i++){
			for(j=1; j<M-1; j++){
				uNew[i][j] = (u[i-1][j]+u[i+1][j]+u[i][j-1]+u[i][j+1]-h*h*S(h*i,k*j))/4.0;
				sum += fabs(uNew[i][j] - u[i][j]);
			}
		}	
		error = sum/(N-2)/(M-2);
		
		for(i=1; i<N-1; i++){
			for(j=1; j<M-1; j++){
				u[i][j] = uNew[i][j];
			}
		}
		++iterations;
//               fprintf(ferror,"%d\t%.7lf\n",iterations,error);
	}
        double end_time=omp_get_wtime();	
//        fclose(ferror);
	
//	print(uNew);
        
        printf("elapsed time: %lf\n",end_time-start_time);
	printf("# of iterations: %d\n",iterations);
	printf("Central value = %lf\n",u[N/2][M/2]);
	
	
/*        FILE *file =  fopen("Poisson-Liebmann.txt", "w");
        
        for(i=0; i<N; i++){
			for(j=0; j<M; j++){
				fprintf(file,"%lf\t",u[i][j]);
			}
			fprintf(file,"\n");
		}
        
        fclose(file);
*/
	return 0;
}
