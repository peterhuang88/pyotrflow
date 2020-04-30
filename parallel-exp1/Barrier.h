#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <pthread.h>

#ifndef BARRIER_H
#define BARRIER_H

typedef struct {
  pthread_mutex_t countLock;
  pthread_cond_t okToProceed;
  int count;
} barrier_t;

class Barrier {
    public:
        Barrier();

        // exec barrier
        void barrier_exec(int num_threads);
    private:
      barrier_t * b;
};

#endif