#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int *reserva_matriz(int tam)
{
    return (int *)malloc(tam * tam * sizeof(int));
}

void free_matriz(int *mat)
{
    free(mat);
}

void rellenar_matriz(int *mat, int tam)
{
    for (int i = 0; i < tam * tam; i++)
        mat[i] = (rand() % 10) + 1;
}

void multiplicar_filas(int *local_C, int *local_A, int *B, int tam, int num_rows)
{
    for (int i = 0; i < num_rows; i++)
        for (int j = 0; j < tam; j++)
        {
            local_C[i * tam + j] = 0;
            for (int k = 0; k < tam; k++)
                local_C[i * tam + j] += local_A[i * tam + k] * B[k * tam + j];
        }
}

int comprobar_argumentos(int argc, char **argv, int world_size, int *tam)
{
    if (argc != 2)
    {
        printf("Uso: mpirun -n <nodos> %s <TAM>\n", argv[0]);
        return 0;
    }

    *tam = atoi(argv[1]);

    if (*tam <= 0)
    {
        printf("Error: TAM debe ser mayor que 0.\n");
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

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int tam;
    int valid = 0;

    if (world_rank == 0)
        valid = comprobar_argumentos(argc, argv, world_size, &tam);

    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!valid)
    {
        MPI_Finalize();
        return 1;
    }

    MPI_Bcast(&tam, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_process = tam / world_size;

    int *A = NULL;
    int *B = reserva_matriz(tam);
    int *C = NULL;
    int *local_A = (int *)malloc(rows_per_process * tam * sizeof(int));
    int *local_C = (int *)malloc(rows_per_process * tam * sizeof(int));

    if (!B || !local_A || !local_C)
    {
        printf("[Nodo %d] Error: fallo de asignacion de memoria.\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (world_rank == 0)
    {
        A = reserva_matriz(tam);
        C = reserva_matriz(tam);

        if (!A || !C)
        {
            printf("[Nodo 0] Error: fallo de asignacion de memoria para A o C.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        srand(time(NULL));
        rellenar_matriz(A, tam);
        rellenar_matriz(B, tam);
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

        free_matriz(A);
        free_matriz(C);
    }

    free_matriz(B);
    free(local_A);
    free(local_C);

    MPI_Finalize();
    return 0;
}