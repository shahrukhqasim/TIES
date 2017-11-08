import os

class FolderDataWriter:
    def __init__(self, path, writer):
        """
        Initiates FolderDataWriter object. Represents a folder data. You can give it path of the folder.

        :param path: Path of folder 
        """
        self.__path = path
        # assert(isinstance(loader, LoadInterface))
        self.__writer = writer

    def write_datum(self, _id, datum):
        full_path = os.path.join(self.__path, _id)
        assert(not os.path.isfile(full_path))
        if not os.path.isdir(full_path):
            os.mkdir(full_path)
        self.__writer.write_datum(full_path, datum)