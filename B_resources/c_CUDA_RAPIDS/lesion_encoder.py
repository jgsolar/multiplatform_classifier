

def lesion_site(lesion):
    if lesion <= 0:
        return 0
    else:
        if 12000 > lesion >= 11000:
            return 11
        else:
            if lesion >= 10000:
                lesion = lesion / 10
            if 1000 > lesion >= 100:
                lesion = lesion * 10
            if 100 > lesion >= 10:
                lesion = lesion * 100
            if 10 > lesion >= 1:
                lesion = lesion * 1000
            return int(lesion / (10**3)) % 10


def lesion_type(lesion):
    if lesion <= 0:
        return 0
    else:
        if 11000 > lesion >= 10000:
            lesion = lesion / 10
        if 1000 > lesion >= 100:
            lesion = lesion * 10
        if 100 > lesion >= 10:
            lesion = lesion * 100
        if 10 > lesion >= 1:
            lesion = lesion * 1000
        return int(lesion / (10**2)) % 10


def lesion_sub_type(lesion):
    if lesion <= 0:
        return 0
    else:
        if 11000 > lesion >= 10000:
            lesion = lesion / 10
        if 1000 > lesion >= 100:
            lesion = lesion * 10
        if 100 > lesion >= 10:
            lesion = lesion * 100
        if 10 > lesion >= 1:
            lesion = lesion * 1000
        return int(lesion / (10**1)) % 10


def lesion_code(lesion):
    if lesion <= 0:
        return 0
    else:
        if (lesion >= 12000) or (11000 > lesion >= 10000):
            return 10
        else:
            if 1000 > lesion >= 100:
                lesion = lesion * 10
            if 100 > lesion >= 10:
                lesion = lesion * 100
            if 10 > lesion >= 1:
                lesion = lesion * 1000
            return int(lesion / (10**0)) % 10


def lesion_number(data):
    numLesion = 0
    if data['lesion_1'] > 0:
        numLesion += 1
    if data['lesion_2'] > 0:
        numLesion += 1
    if data['lesion_3'] > 0:
        numLesion += 1
    return numLesion
