from bloodtype import BloodType
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


model = BloodType(20000,
                  death_rate=BloodType.death_rate,
                  birth_rate=BloodType.birth_rate,
                  timesteptype='y')


model.set_weights(weights={
        'O': 9,
        'A': 2,
        'B': 4,
        'AB': 1
    })

model.steps(100)
model.steps(3, bt_mutation='A')
model.steps(3, bt_mutation='B')
# model.steps(50)
model.steps(5, rf_mutation='-')
# model.steps(100)


model.steps(8000)

model.save_states()
# ipdb.set_trace()
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
