import random
from bloodtype import BloodType

from tqdm import tqdm, trange

import ipdb

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
t = trange(30)
for i in t:
    t.set_description("Population Size {}, year {:.1f}".format(
        model.populationsize, model.time))
    t.refresh()
    model.step()
model.print_state()


t = trange(5)
for i in t:
    t.set_description("Population Size {}, year {:.1f}".format(
        model.populationsize, model.time))
    t.refresh()
    model.step(bt_mutation='A', mutations=100)
    model.step(bt_mutation='B', mutations=100)

model.print_state()

t = trange(100)
for i in t:
    t.set_description("Population Size {}, year {:.1f}".format(
        model.populationsize, model.time))
    t.refresh()
    model.step()
model.print_state()

model.set_fitness(fitness={
        'O': 100,
        'A': 1,
        'B': 5,
        'AB': 20
    })
model.set_death_rate(value=0.3)

for i in tqdm(range(3)):
    model.step()
model.print_state()

model.set_fitness(fitness={
        'O': 1,
        'A': 1,
        'B': 1,
        'AB': 1
    })
model.set_death_rate(value=0.01)


t = trange(100)
for i in t:
    t.set_description("Population Size {}, year {:.1f}".format(
        model.populationsize, model.time))
    t.refresh()
    model.step()
model.print_state()


# model.plotSize()
model.plot_size(save=True)
model.plot_sex(ratio=True, save=True)
model.plot_bt_pt(save=True)
model.plot_bt_pt(showRf=True, save=True)
model.plot_age_groups(ratio=False, save=True)
# ipdb.set_trace()
