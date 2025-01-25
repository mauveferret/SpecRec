import math
from mendeleev import get_all_elements

# Constants
epsilon_0 = 8.854187817e-12  # Vacuum permittivity (F/m)
e = 1.602176634e-19  # Elementary charge (C)
a0 = 5.29177210903e-11  # Bohr radius (m)

# Global variables
m = [0, 0]  # Masses of projectile and target
z = [0, 0]  # Atomic numbers of projectile and target
E = 0  # Energy of the projectile
a = 0  # Screening length
Z = 0  # Product of atomic numbers
U = 0  # Potential energy
c = 0  # Curvature parameter
B = 0  # Impact parameter
o = 0  # Scattering angle
e1 = 0  # Reduced energy
R = 0  # Reduced distance
C1, C2, C3, C4, C5 = 0, 0, 0, 0, 0  # Constants for potential
s1, s2, s3, s4 = 0, 0, 0, 0  # Screening function parameters
d1, d2, d3, d4 = 0, 0, 0, 0  # Screening function parameters
pot = 1  # Potential type


elements = get_all_elements()

def element(ta, por):
    global m, z
    
    """
    elements = [
        (1, 1), (2, 1), (3, 1), (4, 2), (3, 2), (6.94, 3), (9.01, 4), (10.81, 5),
        (12.011, 6), (14, 7), (16, 8), (19, 9), (20.18, 10), (20, 10), (22.99, 11),
        (24.305, 12), (26.9815, 13), (28.086, 14), (30.974, 15), (32.066, 16),
        (35.4527, 17), (39.948, 18), (39.098, 19), (40.078, 20), (44.956, 21),
        (47.867, 22), (50.941, 23), (51.996, 24), (54.938, 25), (55.845, 26),
        (58.933, 27), (58.693, 28), (63.546, 29), (65.39, 30), (69.723, 31),
        (72.61, 32), (74.922, 33), (78.96, 34), (79.904, 35), (83.8, 36),
        (85.4678, 37), (87.62, 38), (88.906, 39), (91.224, 40), (92.906, 41),
        (95.94, 42), (97, 43), (101.07, 44), (102.906, 45), (106.42, 46),
        (107.868, 47), (112.411, 48), (114.818, 49), (118.71, 50), (121.76, 51),
        (127.6, 52), (126.904, 53), (131.29, 54), (132.905, 55), (137.327, 56),
        (138.906, 57), (140.116, 58), (140.908, 59), (144.24, 60), (145, 61),
        (150.36, 62), (151.964, 63), (157.25, 64), (158.925, 65), (162.5, 66),
        (164.93, 67), (167.26, 68), (168.934, 69), (173.04, 70), (174.967, 71),
        (178.46, 72), (180.948, 73), (183.84, 74), (186.207, 75), (190.23, 76),
        (192.217, 77), (195.078, 78), (196.967, 79), (200.59, 80), (204.383, 81),
        (207.2, 82), (208.98, 83), (210, 84), (210, 85), (222, 86), (223, 87),
        (226, 88), (227, 89), (232, 90), (231, 91), (238, 92)
    ]
    #m[por], z[por] = elements[ta]
    """
    
    z[por]=next((el for el in elements if el.symbol == ta), None).atomic_number #projectile atomic number
    m[por]=next((el for el in elements if el.symbol == ta), None).atomic_weight #projectile atomic mass
    

def potenc(r0):
    global U, R
    U = Z * 23.0707e-20 * (s1 * math.exp(-d1 * R) + s2 * math.exp(-d2 * R) + s3 * math.exp(-d3 * R) + s4 * math.exp(-d4 * R)) / (R * a)
    return 0

def pric(r0):
    global B
    potenc(r0)
    if (1 - U / E) < 0:
        return 1
    else:
        P = r0 * math.pow(1 - U / E, 0.5)
        B = P / a
        return 0

def criv(r0):
    global c
    c = 2 * (E - U) * r0 / (a * U + Z * 23.0707e-20 * (s1 * d1 * math.exp(-d1 * R) + s2 * d2 * math.exp(-d2 * R) + s3 * d3 * math.exp(-d3 * R) + s4 * d4 * math.exp(-d4 * R)))
    return 0

def res():
    global e1, B, R, o
    be = (C2 + math.pow(e1, 0.5)) / (C3 + math.pow(e1, 0.5))
    A0 = 2 * (1 + C1 * math.pow(e1, -0.5)) * e1 * math.pow(B, be)
    G = (C5 + e1) / ((C4 + e1) * (math.pow(1 + A0 * A0, 0.5) - A0))
    d = (B + c + A0 * (R - B) / (1 + G)) / (R + c) - math.cos(o / 2)
    return d

def approach(E0):
    global E, a, Z, e1, R, q
    E = 1.6021766e-12 * E0 / (1 + m[0] / m[1])
    z0 = math.pow(z[0], 0.5) + math.pow(z[1], 0.5)
    a = 0.8853 * 0.529e-8 / math.pow(z0, 0.666666666)
    if pot == 1 or pot == 2:
        z0 = math.pow(z[0], 0.23) + math.pow(z[1], 0.23)
        a = 0.88534 * 0.529e-8 / z0
    e1 = a * E / (z[0] * z[1] * 23.0707e-20)
    Z = z[0] * z[1]
    x1 = 0
    x2 = 5e-8
    for i in range(1, 41):
        y = (x1 + x2) / 2
        R = y / a
        q = pric(y)
        if q == 0:
            q = criv(y)
            re = res()
            if re > 0:
                x2 = y
            else:
                x1 = y
        if q == 1:
            x1 = y
    y = (x1 + x2) / 2
    return y

def vybit(E0, o2, od):
    global o, En1, dif1
    hi = math.pi - 2 * o2
    o1 = math.atan(m[1] * math.sin(hi) / (m[0] + m[1] * math.cos(hi)))
    if o1 < 0:
        o1 = math.pi + o1
    print(f"Scattering angle: {o1 * 180 / math.pi:.4f} degrees")
    if (m[1] / m[0]) < 1:
        if o2 > ((math.pi / 4) - (math.asin(m[1] / m[0])) / 2):
            E1 = E0 * math.pow((math.cos(o1) + math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1), 2), 0.5)) / (1 + m[1] / m[0]), 2)
        else:
            E1 = E0 * math.pow((math.cos(o1) - math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1), 2), 0.5)) / (1 + m[1] / m[0]), 2)
    else:
        E1 = E0 * math.pow((math.cos(o1) + math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1), 2), 0.5)) / (1 + m[1] / m[0]), 2)
    print(f"Energy loss: {E0 - E1:.0f} eV")
    En1 = E0 - E1
    o = math.atan(math.sin(o1) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(o1) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
    if o < 0:
        o = math.pi + o
    print(f"Scattering angle: {o * 180 / math.pi:.2f} degrees")
    r0 = approach(E0)
    print(f"Approach distance: {r0 * 1e8:.5f} Å")
    q = pric(r0)
    print(f"Impact parameter: {B * a * 1e8:.5f} Å")
    orm = o2 - od / 2
    orp = o2 + od / 2
    if o2 + od / 2 > math.pi / 2 or orm <= 0:
        print("No solution")
        dif1 = -1
    else:
        hi = math.pi - 2 * orm
        o1 = math.atan(m[1] * math.sin(hi) / (m[0] + m[1] * math.cos(hi)))
        if o1 < 0:
            o1 = math.pi + o1
        if (m[1] / m[0]) < 1:
            if orm > (math.pi / 4 - math.asin(m[1] / m[0]) / 2):
                E1 = E0 * math.pow((math.cos(o1) + math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1), 2), 0.5)) / (1 + m[1] / m[0]), 2)
            else:
                E1 = E0 * math.pow((math.cos(o1) - math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1), 2), 0.5)) / (1 + m[1] / m[0]), 2)
        else:
            E1 = E0 * math.pow((math.cos(o1) + math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1), 2), 0.5)) / (1 + m[1] / m[0]), 2)
        o = math.atan(math.sin(o1) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(o1) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
        if o < 0:
            o = math.pi + o
        r0 = approach(E0)
        q = pric(r0)
        p1 = B * a * 1e8
        hi = math.pi - 2 * orp
        o1 = math.atan(m[1] * math.sin(hi) / (m[0] + m[1] * math.cos(hi)))
        if o1 < 0:
            o1 = math.pi + o1
        if (m[1] / m[0]) < 1:
            if orp > (math.pi / 4 - math.asin(m[1] / m[0]) / 2):
                E1 = E0 * math.pow((math.cos(o1) + math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1), 2), 0.5)) / (1 + m[1] / m[0]), 2)
            else:
                E1 = E0 * math.pow((math.cos(o1) - math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1), 2), 0.5)) / (1 + m[1] / m[0]), 2)
        else:
            E1 = E0 * math.pow((math.cos(o1) + math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1), 2), 0.5)) / (1 + m[1] / m[0]), 2)
        o = math.atan(math.sin(o1) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(o1) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
        if o < 0:
            o = math.pi + o
        r0 = approach(E0)
        q = pric(r0)
        p2 = B * a * 1e8
        print(f"Difference: {abs(p1**2 - p2**2):.7f}")
        dif1 = abs(p1**2 - p2**2)

def rassey(incident_symbol, E0, o1, od, target_symbol):
    
    o1 = o1 * math.pi / 180
    od = od * math.pi / 180
    
    global o, En1, En2, dif1, dif2
    global C1, C2, C3, C4, C5, s1, s2, s3, s4, d1, d2, d3, d4

    element(incident_symbol, 0)  # Example: Neon as projectile
    element(target_symbol, 1)  # Example: Tungsten as target
    
    
    if pot == 0:
        s1, s2, s3, s4 = 0.35, 0.55, 0.1, 0
        d1, d2, d3, d4 = 0.3, 1.2, 6, 0
        C1, C2, C3, C4, C5 = 0.6743, 0.009611, 0.005175, 6.314, 10
    elif pot == 1:
        s1, s2, s3, s4 = 0.028171, 0.28022, 0.50986, 0.18175
        d1, d2, d3, d4 = 0.20162, 0.40290, 0.94229, 3.1998
        C1, C2, C3, C4, C5 = 0.99229, 0.011615, 0.0071222, 9.3066, 14.813
    elif pot == 2:
        s1, s2, s3, s4 = 0.190945, 0.473674, 0.335381, 0
        d1, d2, d3, d4 = 0.278544, 0.637174, 1.919249, 0
        C1, C2, C3, C4, C5 = 1.0144, 0.235809, 0.126, 6.9350, 8.3550
    

    E1 = E0 * math.pow((math.cos(o1) + math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1), 2), 0.5)) / (1 + m[1] / m[0]), 2)
    print(f"Energy after scattering: {E1:.0f} eV")
    En1 = E1
    E2 = E0 * math.pow((math.cos(o1 / 2) + math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1 / 2), 2), 0.5)) / (1 + m[1] / m[0]), 4)
    print(f"Double scattering energy: {E2:.0f} eV")
    o = math.atan(math.sin(o1) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(o1) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
    if o < 0:
        o = math.pi + o
    print(f"Scattering angle: {o * 180 / math.pi:.2f} degrees")
    r0 = approach(E0)
    print(f"Approach distance: {r0 * 1e8:.5f} Å")
    q = pric(r0)
    pc1 = B * a * 1e8
    print(f"Impact parameter: {pc1:.5f} Å")
    if o1 - od / 2 <= 0 or o1 + od / 2 >= 180 or ((math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1 + od / 2), 2))) < 0:
        print("No solution")
        dif1 = -1
    else:
        orm = o1 - od / 2
        orp = o1 + od / 2
        E1 = E0 * math.pow((math.cos(orm) + math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(orm), 2), 0.5)) / (1 + m[1] / m[0]), 2)
        o = math.atan(math.sin(orm) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(orm) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
        if o < 0:
            o = math.pi + o
        r0 = approach(E0)
        q = pric(r0)
        p1 = B * a * 1e8
        E1 = E0 * math.pow((math.cos(orp) + math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(orp), 2), 0.5)) / (1 + m[1] / m[0]), 2)
        o = math.atan(math.sin(orp) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(orp) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
        if o < 0:
            o = math.pi + o
        r0 = approach(E0)
        q = pric(r0)
        p2 = B * a * 1e8
        print(f"Difference: {abs(p1**2 - p2**2):.7f}")
        dif1 = abs(p1**2 - p2**2)

    if (m[1] / m[0]) < 1:
        E1 = E0 * math.pow((math.cos(o1) - math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1), 2), 0.5)) / (1 + m[1] / m[0]), 2)
        print(f"Complementary energy: {E1:.0f} eV")
        En2 = E1
        E2 = E0 * math.pow((math.cos(o1 / 2) - math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1 / 2), 2), 0.5)) / (1 + m[1] / m[0]), 4)
        print(f"Complementary double scattering energy: {E2:.0f} eV")
        o = math.atan(math.sin(o1) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(o1) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
        if o < 0:
            o = math.pi + o
        print(f"Complementary scattering angle: {o * 180 / math.pi:.2f} degrees")
        r0 = approach(E0)
        print(f"Complementary approach distance: {r0 * 1e8:.5f} Å")
        q = pric(r0)
        pc2 = B * a * 1e8
        print(f"Complementary impact parameter: {pc2:.5f} Å")

        if o1 - od / 2 <= 0 or o1 + od / 2 >= 180 or ((math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1 + od / 2), 2))) < 0:
            print("No solution")
            dif2 = -1
        else:
            orm = o1 - od / 2
            orp = o1 + od / 2
            if orp >= 180:
                orp = 360 - orp
            E1 = E0 * math.pow((math.cos(orm) - math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(orm), 2), 0.5)) / (1 + m[1] / m[0]), 2)
            o = math.atan(math.sin(orm) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(orm) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
            if o < 0:
                o = math.pi + o
            r0 = approach(E0)
            q = pric(r0)
            p1 = B * a * 1e8
            E1 = E0 * math.pow((math.cos(orp) - math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(orp), 2), 0.5)) / (1 + m[1] / m[0]), 2)
            o = math.atan(math.sin(orp) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(orp) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
            if o < 0:
                o = math.pi + o
            r0 = approach(E0)
            q = pric(r0)
            p2 = B * a * 1e8
            print(f"Complementary difference: {abs(p1**2 - p2**2):.7f}")
            dif2 = abs(p1**2 - p2**2)

            orm = o1 - 0.00001
            orp = o1 + 0.00001
            E1 = E0 * math.pow((math.cos(orm) - math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(orm), 2), 0.5)) / (1 + m[1] / m[0]), 2)
            o = math.atan(math.sin(orm) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(orm) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
            if o < 0:
                o = math.pi + o
            r0 = approach(E0)
            q = pric(r0)
            p1 = B * a * 1e8
            E1 = E0 * math.pow((math.cos(orp) - math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(orp), 2), 0.5)) / (1 + m[1] / m[0]), 2)
            o = math.atan(math.sin(orp) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(orp) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
            if o < 0:
                o = math.pi + o
            r0 = approach(E0)
            q = pric(r0)
            p2 = B * a * 1e8
            print(f"Complementary differential cross-section: {abs((p2 - p1) * (p1 + p2) / (2 * math.sin(o1) * 2e-5)):.7f}")
    else:
        En2 = -1

    orm = o1 - 0.00001
    orp = o1 + 0.00001
    E1 = E0 * math.pow((math.cos(orm) + math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(orm), 2), 0.5)) / (1 + m[1] / m[0]), 2)
    o = math.atan(math.sin(orm) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(orm) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
    if o < 0:
        o = math.pi + o
    r0 = approach(E0)
    q = pric(r0)
    p1 = B * a * 1e8
    E1 = E0 * math.pow((math.cos(orp) + math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(orp), 2), 0.5)) / (1 + m[1] / m[0]), 2)
    o = math.atan(math.sin(orp) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(orp) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
    if o < 0:
        o = math.pi + o
    r0 = approach(E0)
    q = pric(r0)
    p2 = B * a * 1e8
    print(f"Differential cross-section: {abs((p1 - p2) * (p1 + p2) / (2 * math.sin(o1) * 2e-5)):.7f}")
   




E0 = 15000  # Example: 6000 eV
o1 = 140  # Example: 32 degrees
od = 2  # Example: 2 degrees

rassey("Ne", E0, o1, od, "Au")

