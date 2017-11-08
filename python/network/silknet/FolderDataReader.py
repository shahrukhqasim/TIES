import os
import time
import threading
from silknet.LoadInterface import *
import numpy as np
from random import randint


class FolderDataReader:
    __full_paths_sub_directories = []
    __sampling_sub_directories = []

    __lock = threading.RLock()
    __data = []
    __load_epoch = 0
    __epochs = []
    __ids = []
    __cache_size = 10
    __close_now = False
    __closed = False

    def __init__(self, path, loader, cache_size=10):
        """
        Initiates FolderData object. Represents a folder data. You can give it path of the folder. It will initialize
        all values according to subdirectories of that folder.
        
        :param path: Path of folder 
        """
        self.__path = path
        # assert(isinstance(loader, LoadInterface))
        self.__loader = loader
        self.__cache_size = cache_size

    def __load_more(self):
        index = randint(0, len(self.__sampling_sub_directories) - 1)
        datum_path = self.__sampling_sub_directories[index]
        datum_id = os.path.basename(datum_path)
        del(self.__sampling_sub_directories[index])
        datum = self.__loader.load_datum(datum_path)
        self.__data.append(datum)
        self.__epochs.append(self.__load_epoch)
        self.__ids.append(datum_id)
        if len(self.__sampling_sub_directories) == 0:
            self.__sampling_sub_directories = self.__full_paths_sub_directories.copy()
            self.__load_epoch += 1

    def get_next_epoch(self):
        return self.__epochs[0]

    def next_batch(self, batch_size):
        check = False
        while not check:
            with self.__lock:
                check = len(self.__data) >= batch_size
            continue

        with self.__lock:
            assert(len(self.__data) != 0)

            # Get the values from the lists
            data_return = self.__data[0:batch_size]
            epochs = self.__epochs[0:batch_size]
            ids = self.__ids[0:batch_size]

            # Trim the lists
            self.__data = self.__data[batch_size:]
            self.__epochs = self.__epochs[batch_size:]
            self.__ids = self.__ids[batch_size:]

            # Return values
            return data_return, epochs, ids

    def next_element(self):
            data_return, epochs, ids = self.next_batch(1)

            # Return values
            return data_return[0], epochs[0], ids[0]

    def __worker(self):
        while True:
            with self.__lock:
                if len(self.__data) < self.__cache_size:
                    while len(self.__data) < self.__cache_size:
                        self.__load_more()
                if self.__close_now:
                    self.__closed = True
                    break
            time.sleep(0.02)

    def halt(self):
        self.__closed = False
        self.__close_now = True
        check = False
        while not check:
            with self.__lock:
                check = self.__closed
        return

    def init(self):
        """
        Caches the subdirectories so launches the threads for data loading
        """
        for i in os.listdir(self.__path):
            full_path = os.path.join(self.__path, i)
            self.__full_paths_sub_directories.append(full_path)

        self.__sampling_sub_directories = self.__full_paths_sub_directories.copy()

        while len(self.__data) < self.__cache_size:
            self.__load_more()

        check_and_load_thread = threading.Thread(target=self.__worker, args=())
        self.__close_now = False
        check_and_load_thread.start()