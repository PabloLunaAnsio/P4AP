#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_TAM 1024

//Funcion encargada de rellenar la matriz
//Argumentos:
// -mat: matriz a rellenar en formato 1D
// -tam: tamaño de la matriz (n x n)
//Funcion de tipo void por lo cual no retorna nada
void rellenar_matriz(int *mat, int tam)
{   
    for (int i = 0; i < tam; i++)
        for (int j = 0; j < tam; j++)
            mat[i * tam + j] = (rand() % 10) + 1;
}

//Funcion encargada de multiplicar filas de las matrices
//Argumentos:
// -local_C: matriz resultado local en formato 1D
// -local_A: matriz A local en formato 1D
// -tam: tamaño de la matriz (n x n)
// -num_rows: número de filas que procesa cada nodo
//Funcion de tipo void por lo cual no retorna nada
void multiplicar_filas(int *local_C, int *local_A,int *B, int tam, int num_rows)
{
    for (int i = 0; i < num_rows; i++)
        for (int j = 0; j < tam; j++)
        {
            local_C[i * tam + j] = 0;
            for (int k = 0; k < tam; k++)
                local_C[i * tam + j] += local_A[i * tam + k] * B[k * tam + j];
        }
}
/*
void imprimir_matriz(int mat[MAX_TAM][MAX_TAM], int tam)
{
    for (int i = 0; i < tam; i++)
    {
        for (int j = 0; j < tam; j++)
            printf("%4d ", mat[i][j]);
        printf("\n");
    }
}
*/

//Control de errores para el tamaño que pasamos por línea de argumentos en función número de nodos
//Argumentos:
// -argc: número de argumentos pasados por línea de comandos
// -argv: vector de argumentos pasados por línea de comandos
// -world_size: número de nodos en la ejecución MPI
// -tam: puntero a la variable donde se almacenará el tamaño de la matriz
//La funcion retorna 1 si es correcto el control de errores , 0 si es incorrecto
int comprobar_argumentos(int argc, char **argv, int world_size, int *tam)
{
    if (argc != 2)
    {
        printf("Uso: mpirun -n <nodos> %s <TAM>\n", argv[0]);
        return 0;
    }

    *tam = atoi(argv[1]);

    if (*tam <= 0 || *tam > MAX_TAM)
    {
        printf("Error: TAM debe ser mayor que 0 y menor o igual a %d.\n", MAX_TAM);
        return 0;
    }

    if (*tam % world_size != 0)
    {
        printf("Error: TAM (%d) debe ser multiplo del numero de nodos (%d).\n", *tam, world_size);
        return 0;
    }
    return 1;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int tam;
    int valid = 0;

    valid = comprobar_argumentos(argc, argv, world_size, &tam);

    int A[tam*tam];
    int B[tam*tam];
    int C[tam*tam];

    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!valid)
    {
        MPI_Finalize();
        return 1;
    }

    MPI_Bcast(&tam, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_process = tam / world_size;

    int local_A[tam * tam];
    int local_C[tam * tam];

    //nodo 0 inicializa las matriz
    if (world_rank == 0)
    {
        srand(time(NULL));
        rellenar_matriz(A, tam);
        rellenar_matriz(B, tam);

        printf("Cantidad de Kbytes ocupados por cada una de las matrices: %zu KB\n", tam * tam * sizeof(int) / 1024);
    }

    MPI_Bcast(B, tam * tam, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scatter(A, rows_per_process * tam, MPI_INT,
                local_A, rows_per_process * tam, MPI_INT,
                0, MPI_COMM_WORLD);

    double t_inicio = MPI_Wtime();

    multiplicar_filas(local_C, local_A, B, tam, rows_per_process);

    double t_fin = MPI_Wtime();

    MPI_Gather(local_C, rows_per_process * tam, MPI_INT,
               C, rows_per_process * tam, MPI_INT,
               0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        printf("Tiempo de multiplicacion: %.6f segundos\n", t_fin - t_inicio);
        printf("TAM: %dx%d | Nodos: %d | Filas por nodo: %d\n",
               tam, tam, world_size, rows_per_process);
    }

    MPI_Finalize();
    return 0;
}