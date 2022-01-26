import random
from bloodtype import BloodType

from tqdm import tqdm, trange

import ipdb

# Austria
# A+: 33%
# B+: 6%
# O-: 12%
# O+: 31
# A-:
# B-:
#
# RK
# A+:  37%
# B+:  12%
# O-:   6%
# O+:  30%
# A-:   7%
# B-:   2%
# AB+:  5%
# AB-:  1%

random.seed(1234)
model = BloodType(20000,
                  # deathRate=0.0,
                  # birthRate=0.0,
                  timesteptype='y')

# model.step()
# ipdb.set_trace()
# model.step(bt_mutation='A', mutations=10)
# model.step(bt_mutation='A', mutations=10)
# ipdb.set_trace()


# iterations
model.set_fitness(fitness={
        'O': 15,
        'A': 1,
        'B': 4,
        'AB': 2
    })

model.steps(100)
model.steps(5, bt_mutation='A')
model.steps(5, bt_mutation='B')
model.steps(100)

model.steps(10, rf_mutation='-')
model.steps(1000)

# model.set_fitness(fitness={
#         'O': 1,
#         'A': 1,
#         'B': 2,
#         'AB': 2
#     })

model.set_death_rate(value=2*BloodType.deathRate)
model.steps(5)

model.set_death_rate(value=BloodType.deathRate)
model.set_birth_rate(value=2*BloodType.birthRate)
model.steps(10)

model.set_birth_rate(value=BloodType.birthRate)
model.steps(10)


model.steps(5000)

# model.plotSize()
model.plot_size(save=True)
model.plot_size(cumulativ=True, save=True)
model.plot_sex(save=True)
model.plot_sex(ratio=True, save=True)
model.plot_bt_pt(save=True)
model.plot_bt_pt(showRf=True, save=True)
model.plot_bt_pt(ratio=True, save=True)
model.plot_bt_pt(showRf=True, ratio=True, save=True)
model.plot_age_groups(ratio=False, save=True)
model.plot_age_groups(ratio=True, save=True)
model.save_states()
ipdb.set_trace()
