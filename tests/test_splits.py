from soznet.data.splits import split_patients


def _fake_label_dict():
    return {
        ("patient_1", "ses-1"): [1, 0, 0, 0, 0, 0],
        ("patient_1", "ses-2"): [1, 0, 0, 0, 0, 0],
        ("patient_2", "ses-1"): [0, 1, 0, 0, 0, 0],
        ("patient_2", "ses-2"): [0, 1, 0, 0, 0, 0],
        ("patient_3", "ses-1"): [0, 0, 1, 0, 0, 0],
        ("patient_3", "ses-2"): [0, 0, 1, 0, 0, 0],
        ("patient_4", "ses-1"): [0, 0, 0, 1, 0, 0],
        ("patient_4", "ses-2"): [0, 0, 0, 1, 0, 0],
        ("patient_5", "ses-1"): [0, 0, 0, 0, 1, 0],
        ("patient_5", "ses-2"): [0, 0, 0, 0, 1, 0],
    }


def test_no_patient_leakage():
    label_dict = _fake_label_dict()
    train, val, test = split_patients(label_dict, test_ratio=0.2, val_ratio=0.2, seed=0)

    def patients(keys):
        return {k[0] for k in keys}

    p_train, p_val, p_test = patients(train), patients(val), patients(test)
    assert p_train.isdisjoint(p_val)
    assert p_train.isdisjoint(p_test)
    assert p_val.isdisjoint(p_test)
    assert p_train | p_val | p_test == set(_fake_label_dict() and {k[0] for k in label_dict})
