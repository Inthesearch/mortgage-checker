def exact_match(expected, actual):
    clean_actual = actual.strip().lower().split()
    clean_expected = expected.strip().lower().split()

    return clean_actual == clean_expected

def contains_all(lists, string):
    string_arr = string.lower().strip().split()
    string_arr_joined = " ".join(string_arr)
    list_arr = []
    for item in lists:
        cleaned_item = item.lower().strip().split()
        list_arr.append(" ".join(cleaned_item))
    is_equal = True
    for l in list_arr:
        if l in string_arr_joined:
            continue
        else:
            is_equal = False
            break
    return is_equal

def agreement(human, judge):
    if (len(judge) == 0 and len(human) == 0) or (len(judge) != len(human)):
        raise ValueError("Either the both lists are empty or mismatch in size.")
    elif judge == human:
        return 100, []
    else:
        mismatch = len(judge)
        position = []
        for i in range(len(judge)):
            if judge[i] != human[i]:
                mismatch -= 1
                position.append(i)
        percentage = round(mismatch / len(human) * 100,2)
        return percentage, position

