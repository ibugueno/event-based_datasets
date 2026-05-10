from expelliarmus import Wizard

raw_path = "/media/ignacio/KINGSTON/event-cameras/nefer/event_raw/raw/user_21/user21_2022-06-17_12-13-03.raw"

wiz = Wizard(encoding="evt3")   # o "evt2"
wiz.set_file(raw_path)

events = wiz.read()

t_min = events["t"].min()
t_max = events["t"].max()

print("t_min:", t_min)
print("t_max:", t_max)

print("duración us:", t_max - t_min)
print("duración ms:", (t_max - t_min)/1000)
print("duración s:", (t_max - t_min)/1e6)