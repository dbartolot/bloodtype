import random
from bloodtype import BloodType

from tqdm import tqdm, trange

import ipdb

# Austria
# A+:  37%
# B+:  12%
# O-:   6%
# O+:  30%
# A-:   7%
# B-:   2%
# AB+:  5%
# AB-:  1%

# random.seed(1234)
model = BloodType(20000,
                  death_rate=BloodType.death_rate,
                  birth_rate=BloodType.birth_rate,
                  timesteptype='y')


# iterations
# model.set_fitness(fitness={
#         'O': 8,
#         'A': 2,
#         'B': 4,
#         'AB': 1
#     })
# model.set_fitness(fitness={
#         'O': 7,
#         'A': 2,
#         'B': 4,
#         'AB': 2
#     })

model.steps(100)
model.steps(3, bt_mutation='A')
model.steps(3, bt_mutation='B')
model.steps(100)


model.steps(1000)

#
# model.set_death_rate(value=2*BloodType.death_rate)
# model.steps(5)
#
# model.set_death_rate(value=BloodType.death_rate)
# model.set_birth_rate(value=2*BloodType.birth_rate)
# model.steps(10)
#
# model.set_birth_rate(value=BloodType.birth_rate)
# model.steps(10)
#
# model.steps(20, rf_mutation='-')
#
# model.steps(2000)

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
