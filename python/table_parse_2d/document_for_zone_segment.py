
class ZoneSegmentDocument:
    def __init__(self, input_tensor, classes_tensor, word_mask, zone_mask):
        self.input_tensor = input_tensor
        self.classes_tensor = classes_tensor
        self.word_mask = word_mask
        self.zone_mask = zone_mask