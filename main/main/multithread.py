# This file is meant to optimize run time ny allowing multithreading of the __main__ code
# In a future release we might need to investigate the avantage of multiprocessing in top of multithreading
# A good improvment would be to use the Parralel class defined in sklearn which supports multiprocessing and multithreading

from queue import Queue
from threading import Thread


class MultiThreadCP():
    ''' This local class is used to package all multithreading operations
    during the calibration and prediction process 
    In order to respect the GIL we create a thread and a corresponding queue for each algo '''
    def __init__(self, thread_names=[]):
        self.threading_queues={thread_name:Queue() for thread_name in thread_names}
        for thread_name in thread_names:
            t = Thread(name=thread_name, target=self.worker, args=(thread_name, ))
            t.daemon=True # Allow for the main thread to kill the local thread
            t.start()

    def add_task(self, **task):
        ''' Used to add a task in the threading queue of the corresponding thread, it will then be run automatically '''
        self.threading_queues[task['thread_name']].put(task)

    def worker(self, thread_name):
        ''' Define the function that will be run in each thread '''
        while True:
            task = self.threading_queues[thread_name].get()
            algo=task['algo']
            X_train=task['X_train']
            Y_train=task['Y_train']
            X_test=task['X_test']
            pred_index=task['pred_index']
            algo_cv_params=task['algo_cv_params']  # the remaining elements in the task dict
            algo.calib_predict(X_train, Y_train, X_test, pred_index, **algo_cv_params)
            self.threading_queues[thread_name].task_done()

    def wait(self):
        ''' Allow us to wait for all the thread to finish their job '''
        for _, q in self.threading_queues.items():
            q.join()

    