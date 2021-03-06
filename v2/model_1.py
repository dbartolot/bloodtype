from bloodtype import BloodType
import ipdb


# random.seed(1234)
model = BloodType(20000,
                  death_rate=BloodType.death_rate,
                  birth_rate=BloodType.birth_rate,
                  timesteptype='y')


# iterations

model.steps(100)
model.steps(3, bt_mutation='A')
model.steps(3, bt_mutation='B')
model.steps(100)
model.steps(10, rf_mutation='-')


model.steps(500)

model.set_weights(weights={
        'O': 100,
        'A': 10,
        'B': 10,
        'AB': 1
    })
model.set_death_rate(value=2*BloodType.death_rate)
model.steps(10)

model.set_death_rate(value=BloodType.death_rate)
model.set_weights(weights={
        'O': 100,
        'A': 10,
        'B': 10,
        'AB': 1
    })


model.steps(1000)

# model.plotSize()
model.save_states()
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
ipdb.set_trace()
