PIANO_BARS_CENTER_POS = (325, 425)
PIANO_BAR_A_CENTER = (PIANO_BARS_CENTER_POS[0] - 200, PIANO_BARS_CENTER_POS[1], 'A')
PIANO_BAR_B_CENTER = (PIANO_BARS_CENTER_POS[0] - 100, PIANO_BARS_CENTER_POS[1], 'B')
PIANO_BAR_C_CENTER = (PIANO_BARS_CENTER_POS[0] + 0, PIANO_BARS_CENTER_POS[1], 'C')
PIANO_BAR_D_CENTER = (PIANO_BARS_CENTER_POS[0] + 100, PIANO_BARS_CENTER_POS[1], 'D')
PIANO_BAR_E_CENTER = (PIANO_BARS_CENTER_POS[0] + 200, PIANO_BARS_CENTER_POS[1], 'E')

PIANO_BARS = [PIANO_BAR_A_CENTER, PIANO_BAR_B_CENTER, PIANO_BAR_C_CENTER, PIANO_BAR_D_CENTER, PIANO_BAR_E_CENTER]

def func():
    for x in PIANO_BARS:
        print(x)
        return "done"

func()