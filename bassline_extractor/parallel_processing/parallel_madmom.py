import multiprocessing as mp

class _ParallelProcess(mp.Process):
    """
    task_queue :
        Queue with tasks, i.e. tuples ('processor', 'infile')
    """
    def __init__(self, task_queue, result_queue):
        super(_ParallelProcess, self).__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        
    def run(self):
        """Process all tasks from the task queue and record in the result queue."""
        while True:           
            next_task = self.task_queue.get()
            if next_task is None:
                self.task_queue.task_done()
                break
            else:
                # get the task tuple
                processor, title, track =  next_task
                answer = processor(track)
                # signal that it is done
                self.task_queue.task_done()
                self.result_queue.put((title, answer))
        return
    
def process_batch(processor, files, num_workers):
    
    if num_workers=='auto':
        num_workers=mp.cpu_count()
    print('Num workers: {}'.format(num_workers))

    # create task and result queues
    tasks = mp.JoinableQueue()
    #results = mp.Queue()
    results = mp.SimpleQueue()

    # create working threads
    processes = [_ParallelProcess(tasks, results) for _ in range(num_workers)]
    
    for p in processes: # start the processes
        p.daemon = True
        p.start()
    for title, track in files.items():  # Enqueue jobs
        tasks.put((processor, title, track))

    for i in range(num_workers): # Add a poison pill for each consumer
        tasks.put(None)
    
    tasks.join() # wait for all processing tasks to finish
    
    result_dict = {}
    while not results.empty():
        #result = results.get(block=False)
        result = results.get()
        title, track = result[0], result[1]
        result_dict[title] = track   
    return result_dict