
// que si con 4096x4096 si funciona todo pero te peta con un tamaño de matriz grande, es culpa del docker-compose no es porque se 
// quede sin memoria ram el ordenador
// usar memoria dinamica porque con memoria estatica ocurrelo del stack overflow
// entender la diferencia entre usar un puntero y no un doble puntero
// memoria contigua por eso e smejor usar int *A = (int *)malloc(sizeof(int)*N*N) asi mejor que usar el malloc de INt **A

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        printf("Error a la hora de indicar el numero de argumento. \n");
        exit(0);
    }

    int argumento = atoi(argv[1]);
    printf("%d \n",argumento);

    MPI_Init(NULL, NULL);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); //identifica quien soy dentro de mi cluster
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // me dice el tamaño de mi cluster

    MPI_Finalize();
    return 0;

}