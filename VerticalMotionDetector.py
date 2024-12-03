
class VerticalMotionDetector():
    """
    A class to detect vertical motion of a finger to a bar. If a conservative number (sensitivity) of an increasing
    y-axis position is detected. A true value in is_vertical_motion() function will return. Otherwise, the whole record
    will be removed.

    Every motion detected will be instantly removed from the storage
    """
    def __init__(self, sensitivity_level=3, upward_sensitivity_level=8):
        self.motion_store = dict()
        self.upward_motion_store = dict()
        self.sensitivity_level = sensitivity_level
        self.upward_sensitivity_level = upward_sensitivity_level

    def get_motions(self):
        result = []
        for k, v in self.motion_store.items():
            result.append(f"{k}:{len(v)}")
        return result

    def get_upward_motions(self):
        result = []
        for k, v in self.upward_motion_store.items():
            result.append(f"{k}:{len(v)}")
        return result


    def set_sensitivity(self, value):
        self.sensitivity_level = value

    def get_upward_motion_starting_point(self, finger, bar):
        if bar != "":
            key = f"{bar}_{finger}"
            if key in self.upward_motion_store:
                return self.upward_motion_store[key][0]

        return None

    def store_upward_motion(self, finger, bar, vertical_position):
        if bar != "":
            key = f"{bar}_{finger}"
            if not key in self.upward_motion_store:
                "if the record is not exist, create one"
                self.upward_motion_store[key] = [vertical_position]
                return True
            else:
                "if record exist"
                if len(self.upward_motion_store[key]) < 2:
                    self.upward_motion_store[key].append(vertical_position)
                    return False
                elif vertical_position < self.upward_motion_store[key][-1] or vertical_position < self.upward_motion_store[key][-2]:
                    "check if last action is bigger than the current one before append"
                    self.upward_motion_store[key].append(vertical_position)
                    return True
                else:
                    "The motion is not the same as last two. Remove the whole record"
                    self.upward_motion_store[key] = []
                    return False
        else:
            return False

    def is_upward_vertical_motion(self, finger, bar):
        if finger != None:
            key = f"{bar}_{finger}"
            if key in self.upward_motion_store and len(self.upward_motion_store[key]) >= self.upward_sensitivity_level:
                return True
            else:
                return False
        else:
            # checking the bar only
            finger_set = []
            for i in range(4, 41, 4):
                finger_set.append(f"{bar}_{i}")
            for f in finger_set:
                if f in self.upward_motion_store and len(self.upward_motion_store[f]) >= self.sensitivity_level:
                    return True
            return False

    def store_downward_motion(self, finger, bar, vertical_position):
        if bar != "":
            key = f" downward {bar}_{finger}"
            if not key in self.motion_store:
                "if the record is not exist, create one"
                self.motion_store[key] = [vertical_position]
                return True
            else:
                "if record exist"
                if len(self.motion_store[key]) < 2:
                    self.motion_store[key].append(vertical_position)
                    return True
                elif vertical_position > self.motion_store[key][-1] or vertical_position > self.motion_store[key][-2]:
                    "check if last action is bigger than the current one before append"
                    self.motion_store[key].append(vertical_position)
                    return True
                else:
                    "The motion is not the same as last two. Remove the whole record"
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
