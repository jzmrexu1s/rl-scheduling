class Criticality(object):
    LO = 0
    HI = 1
    

    @staticmethod
    def getCriticalityByStr(value):
        dict = {"LO": 0, "HI": 1}
        return dict[value]