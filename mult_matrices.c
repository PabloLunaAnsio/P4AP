#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_TAM 512

int A[MAX_TAM][MAX_TAM];
int B[MAX_TAM][MAX_TAM];
int C[MAX_TAM][MAX_TAM];

void fill_matrix(int mat[MAX_TAM][MAX_TAM], int tam)
{
    for (int i = 0; i < tam; i++)
        for (int j = 0; j < tam; j++)
            mat[i][j] = (rand() % 10) + 1;
}

void multiply_rows(int local_C[MAX_TAM][MAX_TAM], int local_A[MAX_TAM][MAX_TAM],
                   int tam, int num_rows)
{
    for (int i = 0; i < num_rows; i++)
        for (int j = 0; j < tam; j++)
        {
            local_C[i][j] = 0;
            for (int k = 0; k < tam; k++)
                local_C[i][j] += local_A[i][k] * B[k][j];
        }
}

void print_matrix(int mat[MAX_TAM][MAX_TAM], int tam)
{
    for (int i = 0; i < tam; i++)
    {
        for (int j = 0; j < tam; j++)
            printf("%4d ", mat[i][j]);
        printf("\n");
    }
}

int validate_args(int argc, char **argv, int world_size, int *tam)
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

    if (*tam % 8 != 0)
    {
        printf("Error: TAM (%d) debe ser multiplo de 8.\n", *tam);
        return 0;
    }

    return 1;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int tam;
    int valid = 0;

    if (world_rank == 0)
    {
        valid = validate_args(argc, argv, world_size, &tam);
    }

    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!valid)
    {
        MPI_Finalize();
        return 1;
    }

    MPI_Bcast(&tam, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_process = tam / world_size;

    int local_A[MAX_TAM][MAX_TAM];
    int local_C[MAX_TAM][MAX_TAM];

    if (world_rank == 0)
    {
        srand(time(NULL));
        fill_matrix(A, tam);
        fill_matrix(B, tam);
    }

    MPI_Bcast(B, MAX_TAM * MAX_TAM, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scatter(A, rows_per_process * MAX_TAM, MPI_INT,
                local_A, rows_per_process * MAX_TAM, MPI_INT,
                0, MPI_COMM_WORLD);

    double t_inicio = MPI_Wtime();

    multiply_rows(local_C, local_A, tam, rows_per_process);

    double t_fin = MPI_Wtime();

    MPI_Gather(local_C, rows_per_process * MAX_TAM, MPI_INT,
               C, rows_per_process * MAX_TAM, MPI_INT,
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