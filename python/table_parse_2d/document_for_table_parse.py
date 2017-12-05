
class TableParseDocument:
    def __init__(self, input_tensor, classes_tensor, word_mask, zone_mask):
        self.input_tensor = input_tensor # 256x256x308
        self.classes_tensor = classes_tensor # 256x256x4
        self.word_mask = word_mask # 256x256
        self.zone_mask = zone_mask # 256x256