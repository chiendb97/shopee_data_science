import random


def augment_punct(data, label, num_multiply=5):
    augment_data = []
    augment_label = []

    for address, lbl in list(zip(data, label)):
        if lbl == "/":
            for k in range(num_multiply):
                augment_data.append(address)
                augment_label.append(lbl)

        address = address.replace(",", "")
        lbl = lbl.replace(",", "")
        augment_data.append(address)
        augment_label.append(lbl)

    return augment_data, augment_label


def augment_replace_address(data, label, num_multiply=5):
    augment_data = []
    augment_label = []
    index = [[], [], [], []]

    for idx, info in enumerate(data):
        if info['poi'][0] >= 0 and info['street'][0] >= 0:
            index[0].append(idx)
        elif info['poi'][0] >= 0:
            index[1].append(idx)
        elif info['street'][0] >= 0:
            index[2].append(idx)
        else:
            index[3].append(idx)

    for i, info in enumerate(data):
        if info['poi'][0] >= 0 and info['street'][0] >= 0:
            for idx in random.sample(index[0], min(num_multiply, len(index[0]))):
                poi_start, poi_end = info['poi']
                street_start, street_end = info['street']

                replace_poi_start, replace_poi_end = data[idx]['poi']
                replace_street_start, replace_street_end = data[idx]['street']

                address = info['raw_address']
                replace_address = data[idx]['raw_address']
                new_address = address.copy()

                if poi_start < street_start:
                    new_address[street_start: street_end] = replace_address[replace_street_start: replace_street_end]
                    new_address[poi_start: poi_end] = replace_address[replace_poi_start: replace_poi_end]

                    new_poi_start = poi_start
                    new_poi_end = poi_start + replace_poi_end - replace_poi_start
                    new_street_start = street_start + new_poi_end - poi_end
                    new_street_end = new_street_start + replace_street_end - replace_street_start
                else:
                    new_address[poi_start: poi_end] = replace_address[replace_poi_start: replace_poi_end]
                    new_address[street_start: street_end] = replace_address[replace_street_start: replace_street_end]

                    new_street_start = street_start
                    new_street_end = street_start + replace_street_end - replace_street_start
                    new_poi_start = poi_start + new_street_end - street_end
                    new_poi_end = new_poi_start + replace_poi_end - replace_poi_start

                augment_data.append(
                    {"raw_address": new_address, "poi": (new_poi_start, new_poi_end),
                     "street": (new_street_start, new_street_end)})

                augment_label.append(label[idx])

        elif info['poi'][0] >= 0:
            for idx in random.sample(index[1], min(num_multiply, len(index[1]))):
                poi_start, poi_end = info['poi']
                replace_poi_start, replace_poi_end = data[idx]['poi']

                address = info['raw_address']
                replace_address = data[idx]['raw_address']
                new_address = address.copy()

                new_address[poi_start: poi_end] = replace_address[replace_poi_start: replace_poi_end]
                new_poi_start = poi_start
                new_poi_end = poi_start + replace_poi_end - replace_poi_start

                augment_data.append(
                    {"raw_address": new_address, "poi": (new_poi_start, new_poi_end),
                     "street": (-1, -1)})

                augment_label.append(label[idx])

        elif info['street'][0] >= 0:
            for idx in random.sample(index[2], min(num_multiply, len(index[2]))):
                street_start, street_end = info['street']
                replace_street_start, replace_street_end = data[idx]['street']

                address = info['raw_address']
                replace_address = data[idx]['raw_address']
                new_address = address.copy()

                new_address[street_start: street_end] = replace_address[replace_street_start: replace_street_end]
                new_street_start = street_start
                new_street_end = street_start + replace_street_end - replace_street_start

                augment_data.append(
                    {"raw_address": new_address, "poi": (-1, -1),
                     "street": (new_street_start, new_street_end)})

                augment_label.append(label[idx])

    for k in range(num_multiply):
        for i, info in enumerate(data):
            if info['poi'][0] >= 0 and info['street'][0] >= 0:
                poi_start, poi_end = info['poi']
                street_start, street_end = info['street']
                address = info['raw_address']
                new_address = address.copy()

                if poi_start < street_start:
                    new_address[street_start: street_end] = []
                    new_address[poi_start: poi_end] = []
                else:
                    new_address[poi_start: poi_end] = []
                    new_address[street_start: street_end] = []

                augment_data.append({"raw_address": new_address, "poi": (-1, -1), "street": (-1, -1)})
                augment_label.append("/")

            elif info['poi'][0] >= 0:
                poi_start, poi_end = info['poi']
                address = info['raw_address']
                new_address = address.copy()
                new_address[poi_start: poi_end] = []

                augment_data.append({"raw_address": new_address, "poi": (-1, -1), "street": (-1, -1)})
                augment_label.append("/")

            elif info['street'][0] >= 0:
                street_start, street_end = info['street']
                address = info['raw_address']
                new_address = address.copy()
                new_address[street_start: street_end] = []

                augment_data.append({"raw_address": new_address, "poi": (-1, -1), "street": (-1, -1)})
                augment_label.append("/")

    return augment_data, augment_label
