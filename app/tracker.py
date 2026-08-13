def to_corners(box):
    return float(box[0]), float(box[1]), float(box[0] + box[2]), float(box[1] + box[3])

def union(first, second):
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )

def iou(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    area_first = max(1.0, (first[2] - first[0]) * (first[3] - first[1]))
    area_second = max(1.0, (second[2] - second[0]) * (second[3] - second[1]))
    return overlap / (area_first + area_second - overlap)

def gap(first, second):
    dx = max(0.0, max(first[0], second[0]) - min(first[2], second[2]))
    dy = max(0.0, max(first[1], second[1]) - min(first[3], second[3]))
    return (dx * dx + dy * dy) ** 0.5

def cluster_boxes(boxes, gap_ratio=0.55):
    count = len(boxes)
    parent = list(range(count))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def join(first, second):
        root_first = find(first)
        root_second = find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    corners = [to_corners(box) for box in boxes]
    for i in range(count):
        for j in range(i + 1, count):
            first = corners[i]
            second = corners[j]
            span = max(first[2] - first[0], first[3] - first[1], second[2] - second[0], second[3] - second[1])
            if iou(first, second) > 0.0 or gap(first, second) <= gap_ratio * span:
                join(i, j)
    groups = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())

class PersonTracker:
    def __init__(self, max_missed=20, min_iou=0.15):
        self.max_missed = int(max_missed)
        self.min_iou = float(min_iou)
        self.tracks = {}

    def reset(self):
        self.tracks = {}

    def _next_number(self):
        number = 1
        used = set(self.tracks.keys())
        while number in used:
            number += 1
        return number

    def assign(self, boxes):
        result = [0] * len(boxes)
        if not boxes:
            self._age()
            return result
        groups = cluster_boxes(boxes)
        corners = [to_corners(box) for box in boxes]
        regions = []
        for group in groups:
            region = corners[group[0]]
            for index in group[1:]:
                region = union(region, corners[index])
            regions.append((region, group))
        taken = set()
        pending = []
        for region, group in regions:
            best_number = None
            best_score = self.min_iou
            for number, track in self.tracks.items():
                if number in taken:
                    continue
                score = iou(region, track["region"])
                if score >= best_score:
                    best_score = score
                    best_number = number
            if best_number is None:
                pending.append((region, group))
            else:
                taken.add(best_number)
                self.tracks[best_number] = {"region": region, "missed": 0}
                for index in group:
                    result[index] = best_number
        for region, group in pending:
            number = self._next_number()
            self.tracks[number] = {"region": region, "missed": 0}
            taken.add(number)
            for index in group:
                result[index] = number
        self._age(taken)
        return result

    def _age(self, taken=None):
        taken = taken or set()
        dead = []
        for number, track in self.tracks.items():
            if number in taken:
                continue
            track["missed"] += 1
            if track["missed"] > self.max_missed:
                dead.append(number)
        for number in dead:
            del self.tracks[number]

    def active(self):
        return sorted(self.tracks.keys())