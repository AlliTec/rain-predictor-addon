"""rain_window: leading and trailing edge arrival from a synthetic cell footprint."""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HA_URL", "x")
from rain_predictor import rain_window, rgb_to_rate, _PALETTE, _PALETTE_RATE

LAT0, LON0 = -24.98, 151.86
KX, KY = 111.32 * math.cos(math.radians(LAT0)), 110.57


def block(east_from, east_to, north_from, north_to, step=0.5):
    xs, ys = np.meshgrid(np.arange(east_from, east_to + step / 2, step), np.arange(north_from, north_to + step / 2, step))
    return np.column_stack([LAT0 + ys.ravel() / KY, LON0 + xs.ravel() / KX])


ok = True
def check(label, got, exp):
    global ok
    ok &= got == exp
    print(("PASS" if got == exp else "FAIL") + ":", label, "" if got == exp else f"-> {got!r} expected {exp!r}")

# a cell 20 km long (east-west) from 10 to 30 km WEST of the property, 10 km wide (north-south) across its line, moving east (090) at 30 km/h
cell = block(-30, -10, -5, 5)
w = rain_window(cell, LAT0, LON0, 30, 90)
check("leading edge in 20 min, trailing in 60", (w["start_min"], w["end_min"]), (20, 60))
check("it rains for 40 minutes", w["duration_min"], 40)
check("not raining yet", w["raining_now"], False)
# the same cell already half over the property: rain now, 15 more minutes to go
w = rain_window(block(-15, 7.5, -5, 5), LAT0, LON0, 30, 90)
check("over the property: starts now", (w["start_min"], w["raining_now"]), (0, True))
check("over the property: ends when the trailing edge has passed", w["end_min"], 30)
check("remaining duration is end minus now", w["duration_min"], 30)
check("a cell that has passed gives nothing", rain_window(block(10, 20, -5, 5), LAT0, LON0, 30, 90), None)
check("a cell whose track misses the property by 15 km gives nothing", rain_window(block(-30, -10, 10, 20), LAT0, LON0, 30, 90), None)
check("too slow: nothing", rain_window(cell, LAT0, LON0, 0.5, 90), None)
check("no footprint: nothing", rain_window(None, LAT0, LON0, 30, 90), None)
# a faster cell halves the times
w = rain_window(cell, LAT0, LON0, 60, 90)
check("twice the speed: 10 to 30 minutes", (w["start_min"], w["end_min"], w["duration_min"]), (10, 30, 20))
# heading from the north-west (135) with the cell lined up along that track
th = math.radians(135)
along = block(-30, -10, -5, 5)           # reuse: rotate the block so its long axis points back along the track
a = math.radians(90 - 135)             # turns the east-pointing track onto a 135 degree heading
rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
xy = np.column_stack([(along[:, 1] - LON0) * KX, (along[:, 0] - LAT0) * KY]) @ rot.T
rotated = np.column_stack([LAT0 + xy[:, 1] / KY, LON0 + xy[:, 0] / KX])
w = rain_window(rotated, LAT0, LON0, 30, 135)
check("a diagonal track gives the same times", (w["start_min"], w["end_min"]), (20, 60))
# rain amount: mean rate of the pixels still to arrive x hours of rain x calibration
tens = np.full(len(cell), 10.0)
check("10 mm/h for 40 minutes is 6.7 mm", rain_window(cell, LAT0, LON0, 30, 90, rates=tens)["mm"], 6.7)
check("calibration scales it", rain_window(cell, LAT0, LON0, 30, 90, rates=tens, calibration=2.0)["mm"], 13.3)
check("no rates: no mm", rain_window(cell, LAT0, LON0, 30, 90)["mm"], None)
half = block(-15, 7.5, -5, 5)
check("half over the property: only the 30 min to come", rain_window(half, LAT0, LON0, 30, 90, rates=np.full(len(half), 10.0))["mm"], 5.0)
heavy_tail = np.where(cell[:, 1] < LON0 - 20 / KX, 40.0, 10.0)       # the far half of the cell is heavier
check("heavier rain gives more mm", rain_window(cell, LAT0, LON0, 30, 90, rates=heavy_tail)["mm"] > 6.7, True)
# palette: 54 colours, weakest to strongest, every colour maps to itself, rates only rise
check("palette size", len(_PALETTE), 54)
check("palette rates never fall", bool(np.all(np.diff(_PALETTE_RATE) >= 0)), True)
tile = _PALETTE.astype(np.uint8).reshape(1, -1, 3)
check("each palette colour maps to its own rate", bool(np.allclose(rgb_to_rate(tile)[0], _PALETTE_RATE)), True)
odd = np.array([[[171, 159, 122]]], dtype=np.uint8)           # one step off the weakest tan snaps to it
check("an off-palette colour snaps to the nearest", float(rgb_to_rate(odd)[0, 0]), float(_PALETTE_RATE[0]))
check("tan is drizzle, pink is a downpour", (float(_PALETTE_RATE[0]) < 0.2, float(_PALETTE_RATE[-1]) > 50), (True, True))
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
