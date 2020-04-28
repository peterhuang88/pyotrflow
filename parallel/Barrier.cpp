#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <pthread.h>

#include "Barrier.h"

Barrier::Barrier() {
  barrier_t *b;
  b->count = 0;
  pthread_mutex_init(&(b->countLock), NULL);
  pthread_cond_init(&(b->okToProceed), NULL);
  this->b = b;
}

/**************** PARALLEL FUNCTIONS *******************************/
void Barrier::barrier_exec(int num_threads) {
  pthread_mutex_lock(&(this->b->countLock));
  this->b->count++;
  if(this->b->count == num_threads) {
    this->b->count = 0;
    pthread_cond_broadcast(&(this->b->okToProceed));
  } else {
    while(pthread_cond_wait(&(this->b->okToProceed), &(this->b->countLock)) != 0);
  }
  pthread_mutex_unlock(&(this->b->countLock));
}