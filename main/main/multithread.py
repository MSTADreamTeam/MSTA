# This file is meant to optimize run time ny allowing multithreading of the __main__ code
# In a future release we might need to investigate the avantage of multiprocessing in top of multithreading


from queue import Queue
from threading import Thread


class MultiThreadCP():
    ''' This local class is used to package all multithreading operations
    during the calibration and prediction process '''
    def __init__(self, max_threads=10):
        self.threading_queue=Queue()
        for i in range(max_threads):
            t = Thread(target=self.worker)
            t.setDaemon(True) # Check online to understand exaclty the deamon property
            t.start()

    def add_task(self, **task):
        ''' Used to add a task in the threading queue, it will then be run automatically '''
        self.threading_queue.put(task)

    def worker(self):
        ''' Define the function that will be run in each thread '''
        while True:
            task = self.threading_queue.get()
            algo=task.pop('algo')
            X_train=task.pop('X_train')
            Y_train=task.pop('Y_train')
            X_test=task.pop('X_test')
            pred_index=task.pop('pred_index')
            algo_cv_params=task  # the remaining elements in the task dict
            algo.calib_predict(self, X_train, Y_train, X_test, pred_index, **algos_cv_params)
            self.threading_queue.task_done()

    