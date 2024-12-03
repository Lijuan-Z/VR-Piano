
class VerticalMotionDetector():
    """
    A class to detect vertical motion of a finger to a bar. If a conservative number (sensitivity) of an increasing
    y-axis position is detected. A true value in is_vertical_motion() function will return. Otherwise, the whole record
    will be removed.

    Every motion detected will be instantly removed from the storage
    """
    def __init__(self, sensitivity_level=15):
        self.motion_store = dict()
        self.sensitivity_level = sensitivity_level

    def get_motions(self):
        result = []
        for k, v in self.motion_store.items():
            result.append(f"{k}:{len(v)}")
        return result

    def set_sensitivity(self, value):
        self.sensitivity_level = value

    def store_motion(self, finger, bar, vertical_position):
        if bar != "":
            key = f"{bar}_{finger}"
            if not key in self.motion_store:
                "if the record is not exist, create one"
                self.motion_store[key] = [vertical_position]
                return True
            else:
                "if record exist"
                if len(self.motion_store[key]) == 0:
                    self.motion_store[key].append(vertical_position)
                    return True
                elif vertical_position > self.motion_store[key][-1]:
                    print("add vp")
                    "check if last action is bigger than the current one before append"
                    self.motion_store[key].append(vertical_position)
                    return True
                else:
                    "The motion is not the same as last one. Remove the whole record"
                    self.motion_store[key] = []
                    return False
        else:
            return False


    def is_vertical_motion(self, finger, bar):
        if finger != None:
            key = f"{bar}_{finger}"
            if key in self.motion_store and len(self.motion_store[key]) >= self.sensitivity_level:
                return True
            else:
                return False
        else:
            # checking the bar only
            finger_set = []
            for i in range(4, 41, 4):
                finger_set.append(f"{bar}_{i}")
            for f in finger_set:
                if f in self.motion_store and len(self.motion_store[f]) >= self.sensitivity_level:
                    return True
            return False

    def __str__(self):
        result = ""
        for k, v in self.motion_store.items():
            result += f"{k}:{len(v)}, "
        return result
