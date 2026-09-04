"""Small timestamp caches used by the 5 Hz recorder."""

from collections import deque


class TimeCache:
    def __init__(self, maxlen=512, consumable=False):
        self._items = deque(maxlen=maxlen)
        self._consumable = consumable
        self._consumed = set()
        self._next_key = 0

    def append(self, timestamp, value):
        key = self._next_key
        self._next_key += 1
        self._items.append((key, float(timestamp), value))
        return key

    def nearest(self, timestamp, timeout):
        candidates = [
            item for item in self._items
            if (not self._consumable or item[0] not in self._consumed)
            and abs(item[1] - timestamp) <= timeout
        ]
        if not candidates:
            return None
        best = min(candidates, key=lambda item: (abs(item[1] - timestamp), item[0]))
        if self._consumable:
            self._consumed.add(best[0])
        return best[1], best[2]

    def latest(self):
        return None if not self._items else self._items[-1][1:]
