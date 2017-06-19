import h5py
import tensorflow as tf
import numpy as np
import Queue
import time
import multiprocessing
import threading
import Queue
class DataFlowStatus(object):
    """ Data Flow Status

    Simple class for recording how many data have been processed.

    """

    def __init__(self, batch_size, n_samples):
        self.step = 0
        self.epoch = 0
        self.current_iter = 0
        self.batch_size = batch_size
        self.n_samples = n_samples

    def update(self):
        self.step += 1
        self.current_iter = min(self.step * self.batch_size, self.n_samples)

        if self.current_iter == self.n_samples:
            self.epoch += 1
            self.step = 0
            return True
        return False
    def reset(self):
        self.step = 0
        self.epoch = 0             
class HDF5DataFlow:
    def __init__(self,datalist,coord,epoch =300,shuffle = True, batch_size=2000, num_threads=8,
                 max_queue=2):
        self.coord =coord
        self.batch_size = batch_size
        self.data_list=datalist
        self.shuffle =shuffle
        self.batch_ids_queue = multiprocessing.Queue(maxsize=max_queue)
        self.batch_data_queue= multiprocessing.Queue(maxsize=max_queue)
        self.num_threads=num_threads
        self.n_samples=len(datalist[0])
        assert min([len(v) for v in datalist]) == max([len(v) for v in datalist]) 
        self.data_status = DataFlowStatus(self.batch_size, self.n_samples)
        self.batch_index =self.generate_idx(shuffle)
        self.batch_num_now = -1
    def start(self, reset_status=True):
        """ start.
    
        Arguments:
            reset_status: `bool`. If True, `DataStatus` will be reset.
    
        Returns:   
        """
        # Start to process data and fill queues
        self.clear_queues()
        self.interrupted = False
        # Reset Data Status
        if reset_status:
            self.data_status.reset()
        # Only a single thread needed for batches ids
        bi_threads = [multiprocessing.Process(target=self.fill_batch_ids_queue,name='fill ids')]
        # Multiple threads available for feed batch pre-processing
        fd_threads = [multiprocessing.Process(target=self.fill_batch_data_queue,name='fill data %d'%i)
                      for i in range(self.num_threads)]
        self.threads = bi_threads + fd_threads
        for t in self.threads:
            t.daemon = True
            t.start() 
    def interrupt(self):
    # Send interruption signal to processing queue
        self.interrupted = True
        self.clear_queues()
    def generate_idx(self,shuffle):
        return np.random.permutation(self.n_samples) if shuffle else np.arange(self.n_samples)
    def next(self,timeout=None):
        self.data_status.update()
        return self.batch_data_queue.get(timeout=timeout)        
    def next_batch_ids(self):
        self.batch_num_now += 1
        if self.batch_num_now * self.batch_size +self.batch_size> self.n_samples:
            self.batch_index = self.generate_idx(self.shuffle)  
            self.batch_num_now = 0
        return self.batch_index[self.batch_num_now*self.batch_size:self.batch_num_now*self.batch_size+self.batch_size]        
    def fill_batch_ids_queue(self):
        print 'start fill ids'
        while not self.coord.should_stop() and not self.interrupted:
            ids = self.next_batch_ids()
            if ids is False:
                break
            self.batch_ids_queue.put(ids)
    def fill_batch_data_queue(self):
        print 'start fill data'
        while not self.coord.should_stop() and not self.interrupted:
            batch_ids = self.batch_ids_queue.get()
            if batch_ids is False:
                break      
            if len(batch_ids) == 0:
                print (batch_ids,self.batch_num_now)
            data = self.read_data(batch_ids)
            self.batch_data_queue.put(data)
    def read_data(self,batch_ids):
        for dataset in self.data_list:
            if type(dataset) == h5py._hl.dataset.Dataset:
                dataset._local.astype =None
        return tuple([v[i] for i in batch_ids] for v in self.data_list)
    def clear_queues(self):
        """ clear_queues.
    
        Clear queues.
    
        """
        while not self.batch_data_queue.empty():
            self.batch_data_queue.get()
        while not self.batch_ids_queue.empty():
            self.batch_ids_queue.get()


class HDF5DividerFlow:
    def __init__(self,datalist,coord,divided_parts,epoch =300,shuffle = True, batch_sample_num = 2000,batch_size=1, num_threads=1,
                 max_queue=40):
        self.coord =coord
        self.batch_size = batch_size
        self.batch_samplenum =  batch_sample_num
        self.data_list=datalist
        self.shuffle =shuffle
        self.batch_ids_queue = Queue.Queue(maxsize=max_queue+5)
        self.batch_data_queue= Queue.Queue(maxsize=max_queue)
        self.num_threads=num_threads
        self.n_samples=divided_parts
        self.N = len(datalist[0])
        assert min([len(v) for v in datalist]) == max([len(v) for v in datalist]) 
        self.data_status = DataFlowStatus(self.batch_size, int(self.n_samples*self.N/self.batch_samplenum))
        self.batch_index =self.generate_idx(shuffle)
        self.parts_index = np.linspace(0,self.N,divided_parts+1).astype(int)
        self.batch_num_now = -1
    def start(self, reset_status=True):
        """ start.
    
        Arguments:
            reset_status: `bool`. If True, `DataStatus` will be reset.
    
        Returns:   
        """
        # Start to process data and fill queues
        self.clear_queues()
        self.interrupted = False
        # Reset Data Status
        if reset_status:
            self.data_status.reset()
        # Only a single thread needed for batches ids
        bi_threads = [threading.Thread(target=self.fill_batch_ids_queue)]
        # Multiple threads available for feed batch pre-processing
        fd_threads = [threading.Thread(target=self.fill_batch_data_queue)
                      for i in range(self.num_threads)]
        self.threads = bi_threads + fd_threads
        for t in self.threads:
            t.daemon = True
            t.start()    
    def interrupt(self):
    # Send interruption signal to processing queue
        self.interrupted = True
        self.clear_queues()
    def generate_idx(self,shuffle):     
        return np.random.permutation(self.n_samples) if shuffle else np.arange(self.n_samples)
        
    def next(self,timeout=None):
        self.data_status.update()
        print 'qsize:',self.batch_data_queue.qsize()
        return self.batch_data_queue.get(timeout=timeout)    
    def next_batch_ids(self):
        self.batch_num_now += 1
        if self.batch_num_now * self.batch_size +self.batch_size> self.n_samples:
            self.batch_index = self.generate_idx(self.shuffle)  
            self.batch_num_now = 0
        return self.batch_index[self.batch_num_now*self.batch_size:self.batch_num_now*self.batch_size+self.batch_size]        
    def fill_batch_ids_queue(self):
        while not self.coord.should_stop() and not self.interrupted:
            ids = self.next_batch_ids()
            if ids is False:
                break
            self.batch_ids_queue.put(ids)
    def fill_batch_data_queue(self):
        while not self.coord.should_stop() and not self.interrupted: 
            data= []
            batch_ids = self.batch_ids_queue.get()
            if batch_ids is False:
                break      
            data = self.read_data(batch_ids)
            N=len(data[0])
            idxArr = np.random.permutation(N)
            batch_idx = int(N / self.batch_samplenum)
            for idx in range(batch_idx):
                interval = range(idx*self.batch_samplenum , (idx+1)*self.batch_samplenum)         
                self.batch_data_queue.put([v[idxArr[interval]] for v in data])
                print idx,'put in data queue'
    def read_data(self,parts_id):
        print 'reading !'
        for dataset in self.data_list:
            if type(dataset) == h5py._hl.dataset.Dataset:
                dataset._local.astype =None
        return tuple(v[self.parts_index[parts_id[0]]:self.parts_index[parts_id[0]+1]] for v in self.data_list)
    def clear_queues(self):
        """ clear_queues.
    
        Clear queues.
    
        """
        while not self.batch_data_queue.empty():
            self.batch_data_queue.get()
        while not self.batch_ids_queue.empty():
            self.batch_ids_queue.get()



        
if __name__ == '__main__':
    hread =h5py.File('/media/ltp/40BC89ECBC89DD32/souhu_fusai/train_sentence_fishervectors.h5','r')
    coord = tf.train.Coordinator()
    #d1=h5py.File('debug1.h5','r')['feature']
    #d2=h5py.File('debug2.h5','r')['feature']
    dataset=hread['feature']
    dataflow_1 = HDF5DividerFlow([dataset],coord,divided_parts=20,num_threads=2)
    dataflow_1.start()
    sss= time.time()
    for v in range(100):
        if v == 0:
            r=dataflow_1.next()
        else:
            r = dataflow_1.next(10)
        print v,type(r),type(r[0])

    print time.time()-sss
    
    
  
    